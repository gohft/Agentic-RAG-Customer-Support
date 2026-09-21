import logging
import pandas as pd
from sqlalchemy import MetaData, Table, Text, create_engine, inspect, text, update
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

from src.utils.my_logging import setup_logging
from src.utils.load_config import load_config

setup_logging()
logger = logging.getLogger("db")

class PostgresConnection:
    def __init__(self, config_path: str = 'config/postgres_db.yaml'):
        # load config
        self.config_path = config_path
        self.config = load_config(self.config_path)
        logger.info(f"Configuration loaded from {self.config_path}")

        # establish connection with postgresDB
        db_cfg = self.config["database"]

        url = URL.create(
            drivername=db_cfg["drivername"],
            username=db_cfg["username"],
            password=db_cfg["password"],
            host=db_cfg["host"],
            port=db_cfg["port"],
            database=db_cfg["database"],
        )
        # postgres connection, ping before connecting to ensure connection is good
        # test if connection can be established early on using trivial query
        self.engine = create_engine(url, pool_pre_ping=True)

        with self.engine.connect() as conn:
            # raise error if fail
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL database '{db_cfg['database']}'.")

        # table properties to insert
        # extra column record response from agent
        table_cfg = self.config["table"]
        self.primary_column = table_cfg["primary_column"]
        # list of extra columns
        self.extra_columns = table_cfg["extra_columns"]
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if table exist in database."""
        return inspect(self.engine).has_table(table_name)


    def insert_table(self, data_csv: str, table_name: str):
        """
        Load a CSV into a new table and add three empty columns.
        If the table already exists, insert nothing.
        """
        if self._table_exists(table_name):
            logger.info(f"Table '{table_name}' already exists. No data inserted.")
            return 
        
        try:
            df = pd.read_csv(data_csv)
        # catch error, like empty file or error in csv format
        except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            logger.error(f"Error reading CSV '{data_csv}': {e}")
            return

        # check primary key column is non null and unique 
        pk = df[self.primary_column]
        if pk.isna().any() or pk.duplicated().any():
            logger.error(f"Primary key column '{self.primary_column}' must be unique and non-null.")
            return
        
        # set empty columns for extra columns
        for col in self.extra_columns:
            df[col] = None 

        # formt primary key to sql string
        quote = self.engine.dialect.identifier_preparer.quote
        try:
            with self.engine.begin() as conn:
                df.to_sql(
                    table_name,
                    conn,
                    if_exists="fail",
                    index=False,  # no index column
                    dtype={col: Text() for col in self.extra_columns},  # set text type for agent response
                )
                # add primary key for unique identifier for each row and faster lookup
                conn.execute(
                    text(f"ALTER TABLE {quote(table_name)} ADD PRIMARY KEY ({quote(self.primary_column)})")
                )
        except (SQLAlchemyError, ValueError, KeyError) as e:
            logger.error(f"Error inserting dataframe into table '{table_name}': {e}")
            return

        logger.info(f"Successfully loaded data in {data_csv} to {table_name} table in postgresDB.")
        return 

    def delete_table(self, table_name: str):
        """Drop the table if it exists."""

        if not self._table_exists(table_name):
            logger.info(f"Table '{table_name}' does not exist. Nothing deleted.")
            return 

        # convert the table name to SQL string so that no error in table name
        quoted = self.engine.dialect.identifier_preparer.quote(table_name)
        try:
            with self.engine.begin() as conn:
                conn.execute(text(f"DROP TABLE {quoted}"))
        except SQLAlchemyError as e:
            logger.error(f"Error deleting table '{table_name}': {e}")
            return 

        logger.info(f"Table '{table_name}' deleted.")
        return 

    def download_table(self, table_name: str) -> pd.DataFrame | None:
        """Return the table as a pandas DataFrame (None if it doesn't exist)."""
        if not self._table_exists(table_name):
            logger.info(f"Table '{table_name}' does not exist.")
            return 

        return pd.read_sql_table(table_name, self.engine)


    def update_row(
        self,
        table_name: str,
        index: int,  # integer
        updates: dict, # key value pair to update for row 
    ) -> dict:
        """
        Generic row update function to update column values of a row. 
 
        Updates the row whose index column equals `index`, setting each
        {column: value} pair in `updates`. Every key must be an existing column
        of the table (the index column itself cannot be changed).
 
        Returns {"status": "success" | "not_found" | "error", "message": str}
        and never raises, so it is safe to call from a LangGraph node.
        """
        
        if not self._table_exists(table_name):
            return {"status": "error", "message": f"Table '{table_name}' does not exist."}
 
        try:
            # build SQLAlchemy table object using details from the existing engine with autoload_with=self.engine
            # can cache it for faster update 
            table = Table(table_name, MetaData(), autoload_with=self.engine)

            # check that keys to update exist
            invalid = [k for k in updates if k not in table.c]
            if invalid:
                return {
                    "status": "error",
                    "message": f"Column {invalid} not found in table {table_name}."
                }

            # update row statement 
            stmt = update(table).where(table.c[self.primary_column] == index).values(**updates)
            with self.engine.begin() as conn:
                result = conn.execute(stmt)
 
            if result.rowcount == 0:
                return {
                    "status": "error",
                    "message": f"No row with {self.primary_column} as {index} in table {table_name}.",
                }

            # update row success
            return {
                "status": "success",
                "message": f"Updated row with {self.primary_column} as {index} in table {table_name}.",
            }
        except SQLAlchemyError as e:
            return {"status": "error", "message": f"Database error: {e}"}

    def close(self):
        """Close connection to release resources."""
        self.engine.dispose()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # test 
    db = PostgresConnection() 

    db.insert_table("eval_dataset.csv", "my_table")
    logger.info(db.download_table("my_table").head())

    ag = {"agent_classification": "human", "agent_response": "no response"}

    logger.info(db.update_row("my_table", 1, ag))
    logger.info(db.download_table("my_table").head())
    db.delete_table("my_table")
    db.close()

    # ---- LangGraph usage: wrap update_row in a node -------------------------

# update is dictionary 
# from typing import Any, Dict, TypedDict
# from langgraph.graph import StateGraph, END

# db = database_connection()

# class State(TypedDict, total=False):
#     table_name: str
#     index: int
#     updates: Dict[str, Any]
#     db_result: Dict[str, str]

# def update_row_node(state: State) -> dict:
#     result = db.update_row(state["table_name"], state["index"], state["updates"])
#     return {"db_result": result}

# graph = StateGraph(State)
# graph.add_node("update_row", update_row_node)
# graph.set_entry_point("update_row")
# graph.add_edge("update_row", END)
# app = graph.compile()

# out = app.invoke({
#     "table_name": "my_table",
#     "index": 0,
#     "updates": {"value1": "a", "value2": "b"},
# })
# print(out["db_result"])