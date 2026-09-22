import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from src.utils.my_logging import setup_logging
from src.utils.load_config import load_config
from src.database.chroma_connection import ChromaConnection
from src.graph.states_and_classes import SharedState

setup_logging()
logger = logging.getLogger("graph")

  
class CustomerSupportNode:
    """
    Customer support agent that has RAG tool to query company policy.
    Agent received the customer query,  determine if RAG tool necessary, 
    create input query to exceute RAG tool. Craft response to customer query from the
    documents received from RAG. 
    """
    def __init__(self, chroma: ChromaConnection, config_path: str = 'config/graph/customer_support.yaml'):
        # load config
        self.config_path = config_path
        self.config = load_config(self.config_path)
        logger.info(f"Configuration loaded from {self.config_path}")

        # RAG tool pre-defined values
        rag_cfg = self.config["rag"]
        self.collection_name = rag_cfg["collection_name"]
        self.top_k = rag_cfg["top_k"]
        self.num_return_docs = rag_cfg["num_return_docs"]

        # agent config 
        agent_cfg = self.config["agent"]
        # set LLM params
        llm = ChatOllama(
            model=agent_cfg["model"],
            base_url=self.config["ollama_url"],
            **agent_cfg["parameters"],
        )

        # set the chroma connection used by the RAG tool to query the vector database
        self.chroma = chroma
    
        # create ReAct agent with access to the RAG tool
        # no structured output since output is just a string response
        self.agent = create_agent(
            model=llm,
            tools=[self._build_rag_tool()],
            system_prompt=agent_cfg["system_prompt"],
            #response_format=AgentResponse,  # structured output
        )
        logger.info(f"Customer support node created using model: {agent_cfg["model"]}")


    def _build_rag_tool(self):
        """
        Build the RAG tool the agent uses to search the refund/return policy collection.
        Use wrapper to access the class variables for the tool.
        Use @tool decorator to create tool. Return function that execute the query with 
        pre-defined query parameters and LLM only need to input the search query.
        """
        @tool
        def search_refund_return_policy(query: str) -> str:
            """Search the company's return and refund policy documents for information relevant to a customer's query.

            Before responding to a query, use this tool whenever the customer's query is about the
            company's return/refund policy, or that the customer wants to return an item or get a
            refund for a purchase they made, so that the response is grounded in the actual policy.

            Args:
                query: A focused search phrase describing the customer's specific
                    issue and not the full customer message.

            Returns:
                The most relevant policy passages as plain text, or a message
                indicating nothing relevant was found.
            """
            docs = self.chroma.query_collection_with_rerank(
                input_query=query,
                top_k=self.top_k,
                num_return_docs=self.num_return_docs,
                collection_name=self.collection_name,
            )
            # return list of dictionary, docs under "text" key
            if not docs:
                return "No relevant documents found."

            # combine all the documents to string to return
            return "\n\n".join(d["text"] for d in docs)

        return search_refund_return_policy

    def __call__(self, state: SharedState) -> dict:
        """
        Make class callable for class to behave like function.
        """
        try:
            # pass customer query to LLM
            result = self.agent.invoke({
                "messages": [HumanMessage(content=state["customer_query"])],
            })
        except Exception as e:
            # raise error for retry
            logger.error(f"Error occured: {e}")
            raise

        # message contains full conversation history
        # concatenate all documents extracted by the RAG tool across all tool calls
        # since all tool call result are used to generate final response
        # might retrieved repeated docs
        # document under content key
        retrieved_docs = "\n\n".join(
            m.content for m in result["messages"] if isinstance(m, ToolMessage)
        )

        # collect the search query used for every tool call the agent made,
        # in the order the calls were issued
        tool_queries = [
            tc["args"]["query"]
            for m in result["messages"]
            if isinstance(m, AIMessage)
            for tc in m.tool_calls
        ]

        logger.info("Customer support agent response to query done.")

        # update state with response and docs used
        return {
            "agent_response": result["messages"][-1].content,  # final message is response of LLM, no more tool call
            "retrieved_docs": retrieved_docs,
            "tool_queries": tool_queries,
        }


if __name__ == "__main__":

    test_state = {
        "record_id": 1001,
        "customer_query": "If a buyer hasn't received their parcel and the status on the Shopee app shows it as delivered, what steps should they take to request a refund and what are the policies regarding parcel not delivered?",
        "agent_classification": "agent",
        "agent_response": None,
        "retrieved_docs": None,
        "tool_queries": None,
        "db_update_status": None,
        "db_update_message": None
    }

    test_chroma = ChromaConnection()

    test_node = CustomerSupportNode(test_chroma)
    test_resp = test_node(test_state)

    logger.info(f"Response is {test_resp["agent_response"]}")
    logger.info(f"Query used for tool call is {test_resp["tool_queries"]}")
    logger.info(f"Docs retrieved from tool call is {test_resp["retrieved_docs"]}")