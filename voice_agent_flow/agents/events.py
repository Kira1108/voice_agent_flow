from dataclasses import dataclass
from typing import Dict, Optional
from pydantic import BaseModel

@dataclass
class EventType:
    AgentTextStream: str = 'AgentTextStream'
    ToolCallsOutput: str = 'ToolCallsOutput'
    ToolCallsOutputStart:str = 'ToolCallsOutputStart'
    ToolCallResult: str = 'ToolCallResult'
    AgentTextOutput: str = 'AgentTextOutput'
    InferenceFinish: str = 'InferenceFinish'
    AgentHandoff: str = 'AgentHandoff'
    OtherType: str = ''
    StructuredOutput: str = 'StructuredOutput'
    HangupSignal: str = 'HangupSignal'

@dataclass
class AgentEvent:
    status: Optional[str] = None

@dataclass
class AgentTextStream(AgentEvent):
    delta: str = ''

@dataclass
class ToolCallsOutputStart(AgentEvent):
    message: Dict = None

@dataclass
class ToolCallsOutput(AgentEvent):
    message: Dict = None
    
@dataclass
class AgentHandoff(AgentEvent):
    message: Dict = None

@dataclass
class ToolCallResult(AgentEvent):
    message: Dict = None

@dataclass
class AgentTextOutput(AgentEvent):
    message: Dict = None
    
@dataclass
class StructuredOutput(AgentEvent):
    message: BaseModel = None
    
@dataclass
class HangupSignal(AgentEvent):
    message: BaseModel = None

@dataclass
class AgentResult:
    event: AgentEvent = None
    event_type: str = EventType.OtherType
    finish_reason: str = ''
    last_agent_name: str = ''

