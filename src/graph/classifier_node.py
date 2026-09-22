import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from src.utils.my_logging import setup_logging
from src.utils.load_config import load_config
from src.graph.states_and_classes import Classification, SharedState

setup_logging()
logger = logging.getLogger("graph")

class ClassifierNode:
    def __init__(self, config_path: str = 'config/graph/classifier.yaml'):
        # load config
        self.config_path = config_path
        self.config = load_config(self.config_path)
        logger.info(f"Configuration loaded from {self.config_path}")

        self.system_prompt = self.config["system_prompt"]

        llm = ChatOllama(
            model=self.config["model"],
            base_url=self.config["ollama_url"],
            **self.config["parameters"],
        )

        # method is json_schema, to read schema from pydantic class
        # ensure output match the expected schema to update the state of graph
        self.llm_structured = llm.with_structured_output(Classification, method="json_schema")
        logger.info(f"Classification node created using model: {self.config["model"]}")

    def __call__(self, state: SharedState) -> dict:
        """
        Make class callable for class to behave like function. 
        """
        try:
            # take in the customer query and decide who to handle the query
            result = self.llm_structured.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=state["customer_query"]),
            ])
        except Exception as e:
            logger.error(f"Error occured: {e}")
            raise 

        logger.info("Classification of query done.")
        return {"agent_classification": result.target}
     

if __name__ == "__main__":

    test_state = {
        "record_id": 1001,
        "customer_query": "How can a buyer submit a refund request via the Shopee mobile app if the item received is in its original sealed condition??",
        "agent_classification": None,
        "agent_response": None,
        "retrieved_docs": None,
        "tool_queries": None,
        "db_update_status": None,
        "db_update_message": None
    }

    test_node = ClassifierNode()
    test_class = test_node(test_state)

    logger.info(test_class)
    logger.info(type(test_class))
    logger.info(f"Classfication is {test_class["agent_classification"]}")

