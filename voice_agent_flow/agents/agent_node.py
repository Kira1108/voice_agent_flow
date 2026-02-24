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
    """
    AgentNode represents a single agent in the agent flow, encapsulating the model, instructions, tools, and output structure for that agent. 
    It provides a method to create an Agent instance from this configuration.
    
    Atrributes:
    - name: the name of the agent node
    - model: the LLM model used by this agent
    - instruction: the main instruction template for the agent
    - task_cls: the structured task complete class that defines the expected output format
    - step_instruction: optional step specific instruction to be included in the prompt
    - examples: optional example interactions to be included in the prompt, can be a list of strings or a single string
    - tools: the tools available to the agent, represented as a list of tool definitions.
    """
    
    
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

class DoHangUp(BaseModel):
    '''Complete signal for hangup, no more conversation needed.
    When the agent want to end the call actively, return this signal, a structured output to tell the system to end the call.
    '''

@dataclass
class HangUpNode(AgentNode):
    
    name:str = "hangup"
    model:OpenAIChatModel = None
    instruction: str = (
        "You are a voice agent responsible for handling phone call hangups.\n"
        "Based on the conversation history, generate a polite and concise closing statement to end the call."
        )
    task_cls:BaseModel = DoHangUp
    step_instruction:str = (
        "First, generate a closing statement to indicate you are about the end call(You must include ‘goodbye’ or '再见' in the response message).\n"
        "As soon as the user responds to the closing statement and there are no further questions or issues to address, "
        "Create a structured output of type DoHangUp to signal that the call can be ended.\n",
        "Examples are only for reference, pay attention to the context and gereate a proper closing statement message."
        )
    examples:list | str = field(
        default_factory=lambda: [
        '感谢您的接听，祝您生活愉快，再见。',
        'Thank you for your time, goodbye.']
        
    )
    tools: list = field(default_factory=list)
    
        
    