from pathlib import Path
import logging

import pandas as pd
import litellm
import ollama
from pydantic import BaseModel, Field
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer
#from ragas.testset.synthesizers import default_query_distribution

from src.utils.my_logging import setup_logging
from src.utils.load_config import load_config
from src.utils.pdf_processing import chunk_pdf

setup_logging()
logger = logging.getLogger("dataset_generator")

class UnanswerableQueryList(BaseModel):
    queries: list[str] = Field(description="Possible customer queries excluding topics on refunds and returns.")

class TrustedDatasetGenerator:
    """
    Build a golden synthetic Q&A datasets from a PDF using RAGAS.
    Dataset can be used to evaluate RAG. 
    """

    def __init__(self, pdf_filepath: str, config_path: str = 'config/eval/dataset_generator.yaml'):

        # load config file to get config in dictionary
        self.pdf_filepath = Path(pdf_filepath)
        if not self.pdf_filepath.is_file():
            raise FileNotFoundError(f"PDF not found: {self.pdf_filepath}")

        self.config_path = config_path
        self.config = load_config(self.config_path)
        logger.info(f"Configuration loaded from {self.config_path}")

        # ollama client to use client for repeated calls
        self.ollama_url = self.config["ollama_url"]
        self.ollama_client = ollama.Client(self.ollama_url)

        # RAGAS
        # models
        self.generator_model_config = self.config["generator_model"]
        self.embedding_model = self.config["embedding_model"]

        # dataset
        dataset_cfg = self.config["dataset"]
        self.answerable_size = dataset_cfg["answerable_size"]
        self.unanswerable_size = dataset_cfg["unanswerable_size"]
        self.question_type_distribution = list(dataset_cfg["question_type_distribution"].values())

        # chunking config
        self.chunk_cfg = self.config["chunking"]

        # classification_column
        self.classification_column = self.config["classification_column"]

        # set the configuration for the unanswerable model 
        self.unanswerable_model_cfg = self.config["unanswerable_model"]

        # columns to convert
        self.question_column = self.config["question_column"]
        self.response_column = self.config["response_column"]

        # set index 
        self.index_column = self.config["index_column"]

        # config to convert to email format 
        self.email_formatter_question_model_cfg = self.config["email_formatter_question_model"]
        self.email_formatter_response_model_cfg = self.config["email_formatter_response_model"]


    @staticmethod
    def normalize_to_sum_one(values: list[float], tol: float = 1e-9) -> list[float]:
        """
        Ensure a list of floats sums to 1, preserving their relative ratio.
        Scales each by 1/total to ensure the sum is 1, if it is not the case.
        """
        if not values:
            raise ValueError("Input list cannot be empty.")

        if any(v < 0 for v in values):
            raise ValueError("None of the input can be negative")

        total = sum(values)

        if total == 0:
            raise ValueError("Cannot normalize all zeros — ratio is undefined.")

        if abs(total - 1.0) < tol:
            return list(values)

        scale = 1.0 / total
        return [v * scale for v in values]

    def call_llm(
        self,
        model_name: str,
        user_prompt: str,
        system_prompt: str | None = None,
        params: dict| None = None,
        response_format: dict | str | None = None,   # JSON schema dictinary or "json"
    ) -> str:
        """
        Call an Ollama model and return its text response.

        Args:
            model_name: Name of the Ollama model (required).
            user_prompt: User instruction to the model (required).
            system_prompt: System prompt. (Optional, use default if not provided)
            params: Parameter for LLM model in key-value pairs, e.g.
                {"temperature": 0.2, "num_ctx": 8192}. If None, model defaults
                are used. Unknown keys are ignored rather than raising an error.

        Returns:
            str: The model's text response.
        """
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": user_prompt})

        resp = self.ollama_client.chat(
            model=model_name,
            messages=messages,
            options=params,  # None means "use defaults"
            format=response_format
        )
        return resp["message"]["content"]


    def _set_generator_LLM(self):
        """Create generator LLM for the RAGAS TestsetGenerator using Ollama."""

        # use llm_factory since langchain wrapper getting deprecated soon
        # litellm suppport wide range of models
        generator_llm = llm_factory(
            provider="litellm",
            client=litellm.acompletion,  # use acompletion for async call, since generator is async
            api_base=self.ollama_url,  # use default if not provided
            extra_body = {"think": False}, # non thinking mode
            # model name and LLM arguments
            **self.generator_model_config,
        )

        logger.info("All variables of generator LLM:")
        logger.info(vars(generator_llm))  
        return generator_llm

    def generate_ragas_questions(self) -> pd.DataFrame:
        """Generate golden dataset Q&A to evaluate RAG using RAGAS testset generator."""

        # set the generator LLM
        generator_llm = self._set_generator_LLM()

        logger.info("Generator LLM created.")

        # set the hugging face embedding to async
        embeddings = HuggingFaceEmbeddings(
            model=self.embedding_model,
            use_api=False,
            device="cuda",  #  use cuda
        )

        embeddings.is_async = True
        
        # create the RAGAS testset generator 
        generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings)

        logger.info("Testset generator created.")
        # use the default distribution 0.5 SingleHop, 0.25 MultiHopSpecific, 0.25 MultiHopAbstract
        # query_distribution = default_query_distribution(generator_llm)

        # normalize in case it does not sum to 1
        normalized_distribution = self.normalize_to_sum_one(self.question_type_distribution)
        logger.info(f"Question distribution is : {normalized_distribution}")

        # set the query distribution
        # for singlehop, default is entity type which extract nouns, date and ask them
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=generator_llm), normalized_distribution[0]),
            (MultiHopSpecificQuerySynthesizer(llm=generator_llm), normalized_distribution[1]),
            (MultiHopAbstractQuerySynthesizer(llm=generator_llm), normalized_distribution[2]),
        ]

        # chunk the documents to get list of strings 
        chunks_from_document = chunk_pdf(self.pdf_filepath, embedding_model=self.embedding_model, **self.chunk_cfg)

        testset = generator.generate_with_chunks(
            chunks=chunks_from_document,
            testset_size=self.answerable_size,  # number of questions
            query_distribution=query_distribution,
        )
        # convert to dataframe 
        # dataframe has many columns, "user_input" is question and "reference" is response
        df = testset.to_pandas()

        # add column for classification: agent
        df[self.classification_column] = 'agent'

        logger.info(f"RAGAS testset generation completed. Generated {self.answerable_size} Q&A pairs.")

        return df

    def get_unanswerable_df(self) -> pd.DataFrame:
        """
        Use LLM model to generate unanswerable questions in dataframe.
        """
        num_of_qns = self.unanswerable_size

        # set limit for output where each question is limited
        model_params = {
            "temperature": self.unanswerable_model_cfg["parameters"]["temperature"],
            "num_predict": self.unanswerable_model_cfg["parameters"]["num_predict_per_query"] * num_of_qns,
        }

        # get string output 
        # replace the braces 
        output_qns_string = self.call_llm(
            model_name=self.unanswerable_model_cfg["model"], 
            user_prompt=self.unanswerable_model_cfg["user_prompt"].format(NUM_QUERIES=num_of_qns),
            system_prompt = self.unanswerable_model_cfg["system_prompt"].replace("{NUM_QUERIES}", str(num_of_qns)),
            params=model_params,
            response_format=UnanswerableQueryList.model_json_schema(),  
            # provide the expected schema, get schema in dict format from pydantic class
        )

        # parse the LLM output string to UnanswerableQueryList type
        # valid it and raise error if not 
        output_qns_list = UnanswerableQueryList.model_validate_json(output_qns_string)      
        # remove any empty string
        output_qns_list = [q.strip() for q in output_qns_list.queries if q.strip()]

        # create the question column
        df = pd.DataFrame({self.question_column: output_qns_list}) 

        # return an dataframe, add column for "classification" human
        df[self.classification_column] = 'human'

        logger.info(f"Unanswerable questions generation completed. Generated {num_of_qns} questions.")

        return df


    def convert_to_email_format(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        For the dataframe provided, for each row, rewrite the question and response (if available)
        to email format to be coherent with expected questions and response format.
        Add two new columns (email_format suffix) and returns the dataframe.

        """
        for col in (self.response_column, self.question_column):
            if col not in input_df.columns:
                raise KeyError(f"Column '{col}' not found in input_df.")

        df = input_df.copy()

        # set the new columns names and create list to store 
        question_email_col = f"{self.question_column}_email_format"
        response_email_col = f"{self.response_column}_email_format"

        question_emails = []
        response_emails = []
        
        for i, row in df.iterrows():
            # convert to string for question
            question_val = str(row[self.question_column])
            response_val = row[self.response_column]

            # user prompt is the question
            question_email = self.call_llm(
                model_name=self.email_formatter_question_model_cfg["model"], 
                user_prompt=question_val,
                system_prompt = self.email_formatter_question_model_cfg["system_prompt"],
                params=self.email_formatter_question_model_cfg["parameters"],
            )

            # response is empty for unanswerable
            if pd.isna(response_val):
                response_email = None
            else:
                # create user prompt using both the official response and customer query
                input_llm = (
                    f"CUSTOMER QUERY:\n{question_email}\n\n"
                    f"OFFICIAL RESPONSE:\n{response_val}"
                )

                # user prompt is the response
                response_email = self.call_llm(
                    model_name=self.email_formatter_response_model_cfg["model"], 
                    user_prompt=input_llm,
                    system_prompt=self.email_formatter_response_model_cfg["system_prompt"],
                    params=self.email_formatter_response_model_cfg["parameters"],
                )

            question_emails.append(question_email)
            response_emails.append(response_email)

        df[question_email_col] = question_emails
        df[response_email_col] = response_emails

        logger.info(f"Converted questions and responses to email format. Total number of rows is {len(df)}.")
        return df

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate the dataset to evaluate RAG by first using RAGAS to generate the 
        Q&A pairs and use LLM to generate unanswerable questions.
        Next, convert the questions and response to email format to be coherent with actual data.
        Return the dataset in dataframe.
        """
        # get ragas Q&A 
        ragas_df = self.generate_ragas_questions()
       
        # generate unanswerable questions
        unanswerable_df = self.get_unanswerable_df()

        # combine together
        combined_df = pd.concat([ragas_df, unanswerable_df], ignore_index=True)

        # convert question/response to email format
        converted_df = self.convert_to_email_format(combined_df)

        # add index in first column
        final_df = converted_df.copy()
        final_df.insert(0, self.index_column, range(1, len(final_df) + 1))

        logger.info("Generation of dataset completed.")

        return final_df
    

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    policy_pdf = "shoppee_policy.pdf"
    gen = TrustedDatasetGenerator(policy_pdf)

    # generate eval dataset
    output_df = gen.generate_dataset()
    print(output_df.info())
    output_df.to_csv('dataset50.csv', index=False)

    # qna_df = gen.generate_ragas_questions()

    # logger.info(f"Generated dataframe info: {qna_df.info()}")

    # qna_df.to_csv('qna_test.csv', index=False)

    # df1 = pd.read_csv("qna_test.csv")
    # udf = gen.get_unanswerable_df()

    # print(f"Generated dataframe info: {udf.info()}")
    # combined_df = pd.concat([udf, df1], ignore_index=True)

    # print(combined_df.info())

    # combined_df_email = gen.convert_to_email_format(combined_df)
    # combined_df_email.to_csv('combined_df_email.csv', index=False)
    # logger.info(combined_df_email.info())

