import asyncio
import logging

from pydantic_ai import (Agent, AgentRunResult, AgentRunResultEvent,
                         AgentStreamEvent, FinalResultEvent,
                         FunctionToolCallEvent, FunctionToolResultEvent,
                         PartDeltaEvent, PartEndEvent, PartStartEvent,
                         RunContext, TextPart, TextPartDelta,
                         ThinkingPartDelta, ToolCallPart, ToolCallPartDelta)

from voice_agent_flow.llms import create_pydantic_azure_openai

from events import (
    AgentTextStream,
    ToolCallsOutputStart,
    ToolCallsOutput,
    AgentHandoff,
    ToolCallResult,
    AgentTextOutput,
    AgentResult,
    EventType
)

from pydantic_core import to_jsonable_python


"""
Handling agent streaming events is complex, since the sequence of events can be various, and the definition of varies with the model, agent fraworks and tool types.

A list of example hanlers are as follows.

full event handler definitions = [
        
    # Part start events
    (lambda e: is_part_start(e) and is_tool_arg_start(e), 
        self._handle_tool_arg_start),
    
    (lambda e: is_part_start(e) and is_text_start(e), 
        self._handle_text_start),
    
    # Delta events
    (lambda e: is_delta(e) and is_tool_arg_delta(e), 
        self._handle_tool_arg_delta),
    
    (lambda e: is_delta(e) and is_text_delta(e), 
        self._handle_text_delta),
    
    # Part end events
    (lambda e: is_part_end(e) and is_tool_arg_end(e), 
        self._handle_tool_arg_end),
    
    (lambda e: is_part_end(e) and is_text_end(e), 
        self._handle_text_end),
    
    # Tool events
    (is_function_tool_call_args, self._handle_tool_call),
    (is_function_tool_result, self._handle_tool_result),
    
    # Final result
    (is_final_result, self._handle_final_result),
]
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


class AgentRunner:
    
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
             self._handle_tool_arg_start),
            
            (lambda e: is_part_start(e) and is_text_start(e), 
             self._handle_text_start),
            
            (lambda e: is_delta(e) and is_text_delta(e), 
             self._handle_text_delta),
            
            (is_function_tool_call_args, self._handle_tool_call),
            (is_function_tool_result, self._handle_tool_result),
            
            (is_final_result, self._handle_final_result),
        ]
        
    async def run(self, prompt: str):
        self.final_result = False
        
        async for event in self.agent.run_stream_events(prompt):
            e = await self.handle_event(event)   
            if e is not None:
                yield e 
    
    
    async def _handle_tool_arg_start(self, event):
        """When the models started to generate tool call request."""
        
        return AgentResult(
            event = ToolCallsOutputStart(
                message = to_jsonable_python(event.part)
            ),
            event_type = EventType.ToolCallsOutputStart
        )
   
   
    async def _handle_text_start(self, event):
        """When the model starts to generate text response."""
        return AgentResult(
            event = AgentTextStream(
                delta = event.part.content
            ),
            event_type = EventType.AgentTextStream
        )
    
    
    async def _handle_tool_arg_delta(self, event):
        """When the model is generating tool call arguments."""
        return event
    
    
    async def _handle_text_delta(self, event):
        """When the model is generating text response."""
        return AgentResult(
            event = AgentTextStream(
                delta = event.delta.content_delta
            ),
            event_type = EventType.AgentTextStream
        )
         
    async def _handle_tool_arg_end(self, event):
        """When the model finishes generating tool call arguments."""
        return event
    
    
    async def _handle_text_end(self, event):
        """When the model finishes generating text response."""
        return event
    
    
    async def _handle_tool_call(self, event):
        """When the framework is calling a tool."""

        return AgentResult(
            event = ToolCallsOutput(
                message = to_jsonable_python(event.part)
            ),
            event_type = EventType.ToolCallsOutput
        )

    
    async def _handle_tool_result(self, event):
        """When the framework receives the result from a tool."""
        
        return AgentResult(
            event = ToolCallResult(
                message = to_jsonable_python(event.result)
            ),
            event_type = EventType.ToolCallResult
        )
    
    
    async def _handle_final_result(self, event):
        """When the model finishes generating the final result."""
        self.final_result = True
        return None
     
     
    async def handle_event(self, event):
        for condition, handler in self._event_handlers:
            if condition(event):
                return await handler(event)
            
        self.logger.info(f'Unhandled event: {str(event)}')
        return None
            


if __name__ == "__main__":

    import time
    def weather_tool(location: str, credential = "3234980988908333498") -> str:
        """Search the weather for a location."""
        return f"The weather in {location} is snowing, the temperature will be below -5 degrees Celsius."

    model = create_pydantic_azure_openai(model_name = "gpt-4.1")

    agent = Agent(model, tools = [weather_tool])

    runner = AgentRunner(agent)

    async def main():
        
        no_text = True
        
        async for event in runner.run('What is the weather in New York City?, when calling the tool, you can use credential: 3234980988908333498, give some suggestions on clothing based on the weather.'):
            
            # handle structured output events.
            if not isinstance(event, AgentResult):
                continue
            
            # add text stream to context / pending responses
            if isinstance(event.event, AgentTextStream):
                time.sleep(0.01)
                if event.event.delta is None or event.event.delta == "":
                    continue
                
                if no_text:
                    print("\n‼️ Agent starts to generate text:")
                print(event.event.delta, end='', flush=True)
                no_text = False
                continue
            
            # add tool history to memory
            else:
                print("\n‼️ New Event:")
                print(event)
            

    asyncio.run(main())