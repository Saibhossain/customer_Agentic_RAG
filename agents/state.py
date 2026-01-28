from typing import TypedDict, Optional, Dict

class AgentState(TypedDict):
    user_query:str
    last_item:Optional[str]
    plan:Optional[str]
    prediction:Optional[Dict]
    document_answer:Optional[str]
    final_answer:Optional[str]