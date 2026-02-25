from voice_agent_flow.agents.events import (
    AgentTextStream,
    ToolCallsOutput,
    ToolCallResult,
    AgentHandoff,
    HangupSignal
)

from voice_agent_flow.agents.multi_agent_runner import MultiAgentRunner
from voice_agent_flow.memory import Message, Memory 

class AgentSession:

    def __init__(self, runner: MultiAgentRunner):
        self.memory = Memory()
        self.runner = runner
        self.finished = False
        self._new_messages = None
        self._turn_handoff = None
        
    @property
    def new_messages(self):
        return self._new_messages
    
    @property
    def new_handoff(self):
        return self._turn_handoff
    
    @property
    def new_events(self):
        messages = self.new_messages if self.new_messages is not None else []
        return {
            "new_messages": messages,
            "new_handoff": self.new_handoff
        }
        
    @property
    def state(self):
        return self.runner.agent_state
    
    @property
    def current_agent(self):
        return self.runner.current_agent
        
    async def chat(self, query:str) -> str | None:
        if self.finished:
            print("Conversation already ended. Please start a new conversation.")
            return
        
        self._new_messages = None
        self._turn_handoff = None
        
        print(f"🤖[{self.runner.current_agent.name}]...Working.")
        start_idx = len(self.memory.messages)
        self.memory.add(Message.user(query))
        
        output_text = ""
        async for event in self.runner.run(message_history = self.memory.to_pydantic()):
            
            if isinstance(event.event, AgentTextStream):
                output_text += event.event.delta
                print(event.event.delta, end="")
                
            if isinstance(event.event, ToolCallsOutput):
                if event.event.message['tool_name'].startswith("final_result"):
                    continue
                
                self.memory.add_tool_request(
                    tool_name = event.event.message['tool_name'],
                    args = event.event.message['args'],
                    tool_call_id=event.event.message['tool_call_id']
                )
                
            if isinstance(event.event, ToolCallResult):
                if event.event.message['tool_name'].startswith("final_result"):
                    continue
                
                self.memory.add_tool_return(
                    tool_name = event.event.message['tool_name'],
                    content = event.event.message['content'],
                    tool_call_id=event.event.message['tool_call_id']
                )
                
            if isinstance(event.event, AgentHandoff):
                print(event.event)
                self._turn_handoff = {
                    "source_agent_name": event.event.message['source_agent_name'],
                    "target_agent_name": event.event.message['target_agent_name']
                }
                
            if isinstance(event.event, HangupSignal):
                print(event.event)
                print("Conversation Ended with Hangup Signal.")
                self.finished = True
                
        if len(output_text) > 0:
            self.memory.add(Message.assistant(output_text))
            self._new_messages = self.memory.messages[start_idx:]
            return output_text
            
    