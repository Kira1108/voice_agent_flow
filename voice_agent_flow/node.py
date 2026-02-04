from dataclasses import dataclass, field
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic import BaseModel

from pydantic_ai import (
    Agent
)

@dataclass
class AgentNode:
    name:str
    model:OpenAIChatModel
    instruction:str
    example:str
    task_cls:BaseModel
    tools: list = field(default_factory=list)
    
    def create(self) -> Agent:
        prompt = self.instruction.format(example=self.example)
        
        return Agent(
            name = self.name,
            model = self.model, 
            output_type = self.task_cls | str,
            instructions = prompt, 
            tools = self.tools
        
        )


