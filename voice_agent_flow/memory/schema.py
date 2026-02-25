from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

class Message(BaseModel):
    role: Literal['user','assistant','system','tool']
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @classmethod
    def system(cls, 
               content:str):
        return SystemMessage(content=content)
    
    @classmethod
    def user(cls, 
             content:str):
        return UserMessage(content=content)
    
    @classmethod
    def assistant(cls, 
                  content:str):
        return AssistantMessage(content=content)
    
    @classmethod
    def tool_request(cls, 
                     tool_name:str, 
                     args:str, 
                     tool_call_id:str="fake"):
        return ToolRequestMessage(tool_name=tool_name, args=args, tool_call_id=tool_call_id)
    
    @classmethod
    def tool_return(cls, 
                    tool_name:str,
                    content:str, 
                    tool_call_id:str="fake"):
        return ToolReturnMessage(tool_name=tool_name, content=content, tool_call_id=tool_call_id)
    
class SystemMessage(Message):
    role: str = Field("system")
    content: str
    
class UserMessage(Message):
    role: str = Field("user")
    content: str
    
class AssistantMessage(Message):
    role: str = Field("assistant")
    content: str
    
class ToolRequestMessage(Message):
    role: str = Field("assistant")
    tool_name:str
    args: str
    tool_call_id:str
    
class ToolReturnMessage(Message):
    role: str = Field("tool")
    tool_name:str
    content:str
    tool_call_id:str

class Memory(BaseModel):
    messages: list[Message] = Field(default_factory=list)