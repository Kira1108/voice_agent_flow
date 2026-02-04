from typing import Dict

from pydantic import BaseModel
from pydantic_ai import Agent

from voice_agent_flow.node import AgentNode

ENDING_MESSAGE = "感谢您的接听，祝您生活愉快，再见！"

class AgentRunner:
    def __init__(self, 
                 agents:Dict[str, AgentNode], 
                 entry_agent_name: str,
                 ending_message:str = None
        ):
        
        # the multi-agent container
        self.agents = agents  
        
        # entry agent object
        self.entry_agent = self.get_agent(entry_agent_name) 
        
        # set the entry agent as current agent
        self.current_agent = self.entry_agent
        
        # current message history
        self.all_messages = []
        
        self.collected_information = []
        
        self.ending_message = ending_message if ending_message else ENDING_MESSAGE
        
    def get_agent(self, name: str) -> Agent:
        agent_node = self.agents[name]
        return agent_node.create()
    
    def run(self, input_text:str):
        # first run the current agent
        res = self.current_agent.run_sync(input_text, message_history = self.all_messages)
        
        # get the output
        output = res.output
        
        # update message history
        self.all_messages = res.all_messages()
        
        # if string, return directly
        if isinstance(output, str):
            return output
        
        # if base model, run transfer to get next agent
        elif isinstance(output, BaseModel):
            self.collected_information.append(output)
            print("Collected information:", output.model_dump())
            
            target_agent = output.transfer()
            if target_agent == 'end':
                return self.ending_message
            
            self.current_agent = self.get_agent(target_agent)
            
            # rerun next agent
            res = self.current_agent.run_sync(input_text, message_history = self.all_messages)
            
            # update message history
            self.all_messages = res.all_messages()
            
            # get output
            output = res.output
            return output
        
        else:
            raise ValueError("Unsupported output type from agent.")
        
        return "Error"
    
    def show_information(self):
        objs = self.collected_information
        return [obj.model_dump() for obj in objs]
        
    
    
    
    
    
            