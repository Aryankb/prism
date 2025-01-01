from pydantic import BaseModel
from typing_extensions import TypedDict
from typing import List, Dict, Literal, Annotated, Optional
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str
    previous_actions: List[str]

class OverallState(TypedDict):
    question: str
    rational_plan: str
    notebook: Dict[str,str]
    previous_actions: List[str]

class Agent(BaseModel):
    id: int
    sub_task: str                   #schema for agentic sys (sub task)
    description: Dict[str, str]
    prompt: Optional[str]
    input: Dict[str, str]
    extra_input: List[str]
    output: List[str]