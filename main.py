import logging
import uuid

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from src.database.chroma_connection import ChromaConnection
from src.database.postgres_connection import PostgresConnection
from src.graph.build_graph import build_graph
from src.graph.states_and_classes import SharedState
from src.utils.my_logging import setup_logging

# load environment variables
load_dotenv()
setup_logging()
logger = logging.getLogger("graph")

TABLE_NAME = "test_dataset"

def main():
    """
    Execute the customer support workflow with the test dataset customer query.
    Log each execution of workflow in LangFuse for observability.
    """
    # initialize compiled graph
    postgres = PostgresConnection()
    chroma = ChromaConnection()
    graph = build_graph(postgres, chroma)

    # read env vars to create langfuse client instance (no authentication yet)
    langfuse_client = get_client()

    # set the handler to convert each langgraph steps to logs in langfuse
    # uses the client created internally
    langfuse_handler = CallbackHandler()

    # Group this execution under one Langfuse session
    session_id = str(uuid.uuid4())
    logger.info(f"Starting batch run, langfuse session_id={session_id}")

    # download dataset
    try:
        df_query = postgres.download_table(TABLE_NAME)
    except Exception as e:
        logger.error(f"Failed to download table {TABLE_NAME}: {e}")
        postgres.close()
        return

    # execute workflow for each query in dataset
    try:
        for _, row in df_query.iterrows():
            try:
                # get record id
                record_id = row["record_index"]

                # set the state for each query with customer query
                state: SharedState = {
                    "record_id": record_id,
                    "customer_query": row["user_input"],
                    "agent_classification": None,
                    "agent_response": None,
                    "retrieved_docs": None,
                    "tool_queries": None,
                    "db_update_status": None,
                    "db_update_message": None,
                }

                graph.invoke(
                    state,
                    config={
                        "callbacks": [langfuse_handler],
                        "run_name": f"record_{record_id}", # each row one record
                        "metadata": {
                            "langfuse_session_id": session_id, # all rows in one session
                            "record_id": record_id,
                        },
                    },
                )
            except Exception as e:
                logger.error(f"Row failed: {e}")
    finally:
        # close connection
        # ensure all logs are pushed to LangFuse since log are pushed asynchronously
        langfuse_client.flush()
        postgres.close()


if __name__ == "__main__":
    main()
