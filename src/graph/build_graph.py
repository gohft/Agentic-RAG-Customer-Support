from typing import Literal

from sqlalchemy.exc import SQLAlchemyError
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from src.database.chroma_connection import ChromaConnection
from src.database.postgres_connection import PostgresConnection
from src.graph.classifier_node import ClassifierNode
from src.graph.customer_support_node import CustomerSupportNode
from src.graph.database_update_node import create_database_update_node
from src.graph.states_and_classes import SharedState


def check_classification(state: SharedState) -> Literal["database_update", "customer_support"]:
    """Conditional edge in graph tp route to database_update if classified as human, else to customer_support."""
    if state["agent_classification"] == "human":
        return "database_update"
    return "customer_support"


def build_graph(postgres: PostgresConnection, chroma: ChromaConnection) -> CompiledStateGraph:
    """
    Build the customer query workflow graph.

    Customer query is passed to a `classifier` node to determine who should handle the query.
    If it requires human attention, then it is routed to `database_update` node and ends.
    If agent can handle, it is pass to `customer_support` node to get agent response.
    With agent response, it is pass to `database_update` node and ends.

    Each node has retries in case of exception. 
    
    Args:
        postgres: PosgresConnection required to update the database with agent classification and response.
        chroma: ChromaConnection required for the RAG tool of customer_support node.
    """
    normal_retry_policy = RetryPolicy(max_attempts=3)

    builder = StateGraph(SharedState)

    # nodes with name
    builder.add_node("classifier", ClassifierNode(), retry_policy=normal_retry_policy)
    builder.add_node("customer_support", CustomerSupportNode(chroma), retry_policy=normal_retry_policy)
    builder.add_node(
        "database_update",
        create_database_update_node(postgres),  # use default test_dataset table 
        retry_policy = RetryPolicy( retry_on=SQLAlchemyError, max_attempts=5),  # retry when sql error
    )

    # edges
    builder.add_edge(START, "classifier")
    builder.add_conditional_edges("classifier", check_classification)
    builder.add_edge("customer_support", "database_update")
    builder.add_edge("database_update", END)

    return builder.compile()
