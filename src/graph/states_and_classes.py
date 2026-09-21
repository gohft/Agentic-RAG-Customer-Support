from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class Classification(BaseModel):
    """Decide who should handle the customer query. Limit to two options only."""

    target: Literal["human", "agent"] = Field(
        description=(
            "'agent' if the query can be handled by the LLM, "
            "'human' if the query is beyond the scope of the LLM "
            "and should be handled by a human."
        )
    )
    
class SharedState(TypedDict):
    """
    Shared state of the workflow.

    Attributes:
        record_id: The unique identifier for the customer query from the database. 
        customer_query: The customer query extracted from database. 
        agent_classification: Classification of query, to be either "agent" or "human". 
                            Can be None at initial state before classification.
        agent_response: LLM response to query if assigned to "agent". 
                        Can be None at intial state before agent provide response.
        retrieved_docs: Documents extracted by LLM using RAG tool to provide a response to query.
                        Can be None at intial state before agent provide response.
    """
    record_id: int
    customer_query: str
    agent_classification: str | None
    agent_response: str | None
    retrieved_docs: str | None       
