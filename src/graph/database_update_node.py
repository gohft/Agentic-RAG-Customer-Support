import logging
from sqlalchemy.exc import SQLAlchemyError

from src.utils.my_logging import setup_logging
from src.graph.states_and_classes import SharedState
from src.database.postgres_connection import PostgresConnection

setup_logging()
logger = logging.getLogger("graph")

def create_database_update_node(db: PostgresConnection, table_name: str):
    """
    Factory to create the database update node. 
    Node only updates the state of graph without caring about db connection.
    Return function that has fixed postgres connection and will update to fixed table.

    Args:
        db: Instance of connection to PostgreSQL that is connect to db to update.
        table_name: Name of table to update. Table should contain customer query.
    """

    def db_update_node(state: SharedState) -> dict:
        """
        Insert the shared state of graph to database.
        "agent_response" and  "retrieved_docs" might be none if 
        agent is not assigned to handle the query.
        Update the shared state regarding the status of this database update.

        Args:
            state: Current state of the graph.
        
        Returns:
            Dictionary which is partially update state of graph on the db update status.

        Raises:
            SQLAlchemyError if error occurs while trying to insert values to database
            that is not related to missing table/column/primary key.
        """
        # result is dict with 'status' and 'message' 
        # updates is key value pair, with keys in table
        result = db.update_row(
            table_name=table_name,
            index=state["record_id"],
            updates={"agent_response": state["agent_response"],
                     "agent_classification": state["agent_classification"],
                     "retrieved_docs": state["retrieved_docs"]},
        )
        # raise error is sqlalchemy error to allow for retry
        if result["status"] == "sqlalchemy error":
            logger.error("SQLAlchemy error when updating database.")
            raise SQLAlchemyError(result["message"])

        # update the state, else no update after all retries failed
        logger.info("Database status and message updated.")
        return {"db_update_status": result["status"], "db_update_message": result["message"]}

    # return the function
    return db_update_node

