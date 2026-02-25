from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime
from voice_agent_flow.agents.message_adaptor import pmsg
from typing import Any

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
    content: str = Field(...)
    
class UserMessage(Message):
    role: str = Field("user")
    content: str = Field(...)
    
class AssistantMessage(Message):
    role: str = Field("assistant")
    content: str = Field(...)
    
class ToolRequestMessage(Message):
    role: str = Field("assistant")
    tool_name:str = Field(...)
    args: str = Field(...)
    tool_call_id:str = Field(...)
    
class ToolReturnMessage(Message):
    role: str = Field("tool")
    tool_name:str = Field(...)
    content:str = Field(...)
    tool_call_id:str = Field(...)

class Memory(BaseModel):
    messages: list[Any] = Field(default_factory=list)
    
    def add(self, message: Message):
        self.messages.append(message)
        
    def add_user(self, content:str):
        self.add(Message.user(content))
        
    def add_assistant(self, content:str):
        self.add(Message.assistant(content))
        
    def add_system(self, content:str):
        self.add(Message.system(content))
        
    def add_tool_request(self, tool_name:str, args:str, tool_call_id:str="fake"):
        self.add(Message.tool_request(tool_name, args, tool_call_id))
        
    def add_tool_return(self, tool_name:str, content:str, tool_call_id:str="fake"):
        self.add(Message.tool_return(tool_name, content, tool_call_id))
        
    def to_pydantic(self):
        return pmsg.to_history(self.model_dump()['messages'])