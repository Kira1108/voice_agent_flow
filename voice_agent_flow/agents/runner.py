import logging
from typing import AsyncGenerator



from pydantic_ai import (Agent, AgentRunResult, AgentRunResultEvent,
                         AgentStreamEvent, FinalResultEvent,
                         FunctionToolCallEvent, FunctionToolResultEvent,
                         PartDeltaEvent, PartEndEvent, PartStartEvent,
                         RunContext, TextPart, TextPartDelta,
                         ThinkingPartDelta, ToolCallPart, ToolCallPartDelta)

from voice_agent_flow.llms import create_pydantic_azure_openai

from voice_agent_flow.agents.events import (
    AgentTextStream,
    ToolCallsOutputStart,
    ToolCallsOutput,
    AgentHandoff,
    ToolCallResult,
    AgentTextOutput,
    AgentResult,
    EventType,
    StructuredOutput
)

from pydantic_core import to_jsonable_python

from pydantic import BaseModel

"""
Voice Agent Event Adapter for pydantic_ai

This module bridges pydantic_ai's streaming events to our voice agent framework's event schema.

Why Map Events?
---------------
Our voice agent framework has its own event system (defined in the `events` package).
To integrate pydantic_ai, we translate its native events into our framework's events,
allowing the rest of the pipeline (TTS, interruption handling, etc.) to work seamlessly.

Why External Memory Management?
-------------------------------
Traditional text agent frameworks manage conversation history internally—each new query
is automatically appended to memory. However, voice agents have unique requirements:

1. **Partial Playback**: LLM generation is faster than TTS playback. When a user interrupts,
   only the *actually heard* portion should be committed to memory, not the full generated text.

2. **Turn Ordering**: If a new user query arrives mid-generation, we must:
   - Finalize the current (possibly truncated) assistant response first
   - Then append the new user utterance
   
   This ensures memory reflects the *actual conversation* as experienced by the user.

Architecture Decision
---------------------
The voice agent framework owns conversation history. The text agent (pydantic_ai) is treated
as a stateless generation service—we pass it the full, externally-managed history on each turn.

    ┌──────────────────────────────────────┐
    │  Voice Agent Framework (this module) │
    │  - Manages true conversation state   │
    │  - Tracks what user actually heard   │
    │  - Handles interruptions & ordering  │
    └──────────────────┬───────────────────┘
                       │ passes history
                       ▼
    ┌──────────────────────────────────────┐
    │  pydantic_ai Agent (stateless)       │
    │  - Generates responses               │
    │  - Executes tools                    │
    └──────────────────────────────────────┘
    
    
Event Handler Reference
-----------------------
pydantic_ai emits various streaming events. Below is the full handler mapping,
though not all handlers are active in every configuration.

Event Types:
  - PartStartEvent  : A new part (text/tool call) begins
  - PartDeltaEvent  : Incremental content for an in-progress part  
  - PartEndEvent    : A part completes
  - FunctionToolCallEvent  : Tool is being invoked
  - FunctionToolResultEvent: Tool returned a result
  - FinalResultEvent: Generation complete

Handler Matrix:
┌─────────────────────┬──────────────────────┬─────────────────────────┐
│ Event               │ Condition            │ Handler                 │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ PartStartEvent      │ ToolCallPart         │ _handle_tool_arg_start  │
│ PartStartEvent      │ TextPart             │ _handle_text_start      │
│ PartDeltaEvent      │ TextPartDelta        │ _handle_text_delta      │
│ PartDeltaEvent      │ ToolCallPartDelta    │ _handle_tool_arg_delta  │
│ PartEndEvent        │ ToolCallPart         │ _handle_tool_arg_end    │
│ PartEndEvent        │ TextPart             │ _handle_text_end        │
│ FunctionToolCall    │ -                    │ _handle_tool_call       │
│ FunctionToolResult  │ -                    │ _handle_tool_result     │
│ FinalResultEvent    │ -                    │ _handle_final_result    │
│ AgentRunResultEvent │ output is BaseModel  │ _handle_agent_run_result│
└─────────────────────┴──────────────────────┴─────────────────────────┘
"""


is_part_start = lambda event: isinstance(event, PartStartEvent)
is_tool_arg_start =  lambda event: isinstance(event.part, ToolCallPart)
is_text_start = lambda event: isinstance(event.part, TextPart) or isinstance(event.part, TextPartDelta)

is_part_end = lambda event: isinstance(event, PartEndEvent)
is_tool_arg_end = lambda event: isinstance(event.part, ToolCallPart)
is_text_end = lambda event: isinstance(event.part, TextPart)

is_delta = lambda event: isinstance(event, PartDeltaEvent)
is_tool_arg_delta = lambda event: isinstance(event.delta, ToolCallPartDelta)
is_text_delta = lambda event: isinstance(event.delta, TextPartDelta)

is_function_tool_call_args = lambda event: isinstance(event, FunctionToolCallEvent)
is_function_tool_result = lambda event: isinstance(event, FunctionToolResultEvent)

is_final_result = lambda event: isinstance(event, FinalResultEvent)

is_agent_run_result = lambda event: isinstance(event, AgentRunResultEvent)
is_pydantic_model = lambda obj: isinstance(obj, BaseModel)


class SingleAgentRunner:
    
    def __init__(
        self, 
        agent, 
        logger:logging.Logger = None
    ):
        
        self.agent:Agent = agent
        self.final_result = False
        self.logger = logger if logger else logging.getLogger(self.__class__.__name__)
    
        
        self._event_handlers = [
            (lambda e: is_part_start(e) and is_tool_arg_start(e), 
             self.on_tool_arg_start),
            
            (lambda e: is_part_start(e) and is_text_start(e), 
             self.on_text_start),
            
            (lambda e: is_delta(e) and is_text_delta(e), 
             self.on_text_delta),
            
            (is_function_tool_call_args, self.on_tool_call_args),
            (is_function_tool_result, self.on_tool_result),
            
            (is_final_result, self.on_final_result),
            
            (lambda e: is_agent_run_result(e) and is_pydantic_model(e.result.output), self.on_agent_run_result)
        ]
        
        
    async def run(self, 
                prompt: str = None, 
                message_history:list = None) -> AsyncGenerator[AgentResult, None]:
        
        self.final_result = False
        
        async for event in self.agent.run_stream_events(
            prompt, message_history=message_history):
            
            e = await self.handle_event(event)  
            
            if not isinstance(e, AgentResult):
                continue
             
            if e is not None:
                yield e 
    
    
    async def on_tool_arg_start(self, event:PartStartEvent):
        """When the models started to generate tool call request."""
        logging.info("呃, 稍等我想下啊。")
        return AgentResult(
            event = ToolCallsOutputStart(
                message = to_jsonable_python(event.part)
            ),
            event_type = EventType.ToolCallsOutputStart
        )
   
   
    async def on_text_start(self, event:PartStartEvent):
        """When the model starts to generate text response."""
        return AgentResult(
            event = AgentTextStream(
                delta = event.part.content
            ),
            event_type = EventType.AgentTextStream
        )

    
    async def on_text_delta(self, event:PartDeltaEvent):
        """When the model is generating text response."""
        return AgentResult(
            event = AgentTextStream(
                delta = event.delta.content_delta
            ),
            event_type = EventType.AgentTextStream
        )
        
        
    async def on_tool_call_args(self, event:FunctionToolCallEvent):
        """When the framework is calling a tool."""

        return AgentResult(
            event = ToolCallsOutput(
                message = to_jsonable_python(event.part)
            ),
            event_type = EventType.ToolCallsOutput
        )

    
    async def on_tool_result(self, event:FunctionToolResultEvent):
        """When the framework receives the result from a tool."""
        
        return AgentResult(
            event = ToolCallResult(
                message = to_jsonable_python(event.result)
            ),
            event_type = EventType.ToolCallResult
        )
        
    async def on_agent_run_result(self, event:AgentRunResultEvent):
        """if the final output is a pydantic model, return it, otherwise return None. because the text output has been streamed in delta."""
        
        return AgentResult(
            event = StructuredOutput(
                message = event.result.output
            ),
            event_type = EventType.StructuredOutput
        )
        
        
    async def on_final_result(self, event:FinalResultEvent):
        """When the model finishes generating the final result."""
        self.final_result = True
        return None
         
         
    async def on_tool_arg_end(self, event:PartEndEvent):
        """When the model finishes generating tool call arguments."""
        return None
    
    
    async def on_text_end(self, event:PartEndEvent):
        """When the model finishes generating text response."""
        return None
    
    
    async def on_tool_arg_delta(self, event:PartDeltaEvent):
        """When the model is generating tool call arguments."""
        return None
    
     
    async def handle_event(self, event):
        for condition, handler in self._event_handlers:
            if condition(event):
                return await handler(event)
            
        return None