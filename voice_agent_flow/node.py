from dataclasses import dataclass, field
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic import BaseModel

from pydantic_ai import (
    Agent
)

PROMPT_TEMPLATE = """
## Global Instruction:
{global_instruction}

"""

STEP_INSTRUCTION_TEMPLATE = """
## Current Step Instruction:
{step_instruction}

"""

EXAMPLE_INSTRUCTION_TEMPLATE:str = """
## Example Interaction:
{example_interaction}

"""
@dataclass
class AgentNode:
    """the name of the agent node"""
    name:str
    
    """the LLM model used by this agent"""
    model:OpenAIChatModel
    
    """the main instruction template"""
    instruction:str
    
    """the structured task complete class"""
    task_cls:BaseModel
    
    """the step specific instruction"""
    step_instruction:str = None
    
    """the example interactions to include in the prompt"""
    examples:list | str = None
    
    """the tools available to the agent"""
    tools: list = field(default_factory=list)
    
    def __post_init__(self):
        self.full_instruction = self.instruction
        
        if self.step_instruction:
            self.full_instruction += STEP_INSTRUCTION_TEMPLATE.format(
                step_instruction=self.step_instruction
            )
        if self.examples:
            if isinstance(self.examples, str):
                example_str = self.examples
            else:
                example_str = "\n".join(self.examples)
                
            self.full_instruction += EXAMPLE_INSTRUCTION_TEMPLATE.format(
                example_interaction=example_str
            )
    
    def create(self) -> Agent:
        
        return Agent(
            name = self.name,
            model = self.model, 
            output_type = self.task_cls | str,
            instructions = self.full_instruction, 
            tools = self.tools
        )


