"""
Why I have to map the original pydantic_ai events to my own event schema?
Because I already have a voice agent framework, this framework handles events defined by the events package.
To integrate with the framework, I need to convert the events from pydantic_ai into the events defined in my framework, 
so that the rest of the framework can handle them properly.

Next step, I don't want to pass only a query(current turn) to the agent.
The agent frameworks often handle memory internally, the new query is added to the memory automatically.
However, in the voice agent scenario, there are several differences.
1. The genration speed is faster than tts playback speed. Only what user hears shoule be added to the memory.
    When interruption is enabled, the user may stop the agent's response in the middle, at this time, the text agent is unaware of the actual response.
    
2. When a new user query comes in while text is generating. The new query should not be added to the memory until the agent finishes the current response(interrupted or finished playing)
    Then the user utterance shouledbe added following the agent response in the memory, say... finalizing current agent utterance first, then add the user query.
    
In these schenaios, both the order and content might be different from the original agent framework.

So, voice agent shoule manage chatting history and memory in the voice agent framework rather than the text agent framework.
"""

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
    EventType,
    StructuredOutput
)

from pydantic_core import to_jsonable_python

from pydantic import BaseModel

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
is_pydantic_model = lambda obj: isinstance(obj, BaseModel)


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
            
            (lambda e: is_agent_run_result(e) and is_pydantic_model(e.result.output), self._handle_agent_run_result)
        ]
        
    async def run(self, prompt: str):
        self.final_result = False
        
        async for event in self.agent.run_stream_events(prompt):
            e = await self.handle_event(event)   
            if e is not None:
                yield e 
    
    
    async def _handle_tool_arg_start(self, event):
        """When the models started to generate tool call request."""
        print("呃, 稍等我想下啊。")
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

    
    async def _handle_text_delta(self, event):
        """When the model is generating text response."""
        return AgentResult(
            event = AgentTextStream(
                delta = event.delta.content_delta
            ),
            event_type = EventType.AgentTextStream
        )
        
        
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
        
    async def _handle_agent_run_result(self, event):
        """if the final output is a pydantic model, return it, otherwise return None. because the text output has been streamed in delta."""
        
        return AgentResult(
            event = StructuredOutput(
                message = event.result.output
            ),
            event_type = EventType.StructuredOutput
        )
        
        
    async def _handle_final_result(self, event):
        """When the model finishes generating the final result."""
        self.final_result = True
        return None
         
         
    async def _handle_tool_arg_end(self, event):
        """When the model finishes generating tool call arguments."""
        return event
    
    
    async def _handle_text_end(self, event):
        """When the model finishes generating text response."""
        return event
    
    
    async def _handle_tool_arg_delta(self, event):
        """When the model is generating tool call arguments."""
        return event
    
     
    async def handle_event(self, event):
        for condition, handler in self._event_handlers:
            if condition(event):
                return await handler(event)
            
        self.logger.info(f'Unhandled event: {str(event)}')
        return None
            


if __name__ == "__main__":

    import time
    from pydantic import BaseModel
    
    class Person(BaseModel):
        name:str
        age:int
        
        def transfer(self):
            print("\n‼️ I want to transfer to the next step.")
        
        
    def weather_tool(location: str, credential = "3234980988908333498") -> str:
        """Search the weather for a location."""
        return f"The weather in {location} is snowing, the temperature will be below -5 degrees Celsius."
    
    tool_call_query = 'What is the weather in New York City?'
    name_query = "My name is wanghuan, 32 years old."
    

    model = create_pydantic_azure_openai(model_name = "gpt-4.1")
    
    # this is the actual agent definition.
    agent = Agent(
        model, 
        tools = [weather_tool],
        instructions = (
        "You are a helpful assistant. you chat with users friendly."
        "When user asks questions about weather, do call the weather tool to get the weather information, and give suggestions on clothing based on the weather. "
        "When user mentions any person with name and age, you extract the name and age, return a json object defined by `Person` schema."
        "Reply in Chinese."
        ""),
        output_type = str | Person)

    runner = AgentRunner(agent)

    async def run_one_turn(query):
        
        no_text = True
        
        async for event in runner.run(query):
            
            # handle structured output events.
            # if not isinstance(event, AgentResult):
            #     continue
            
            # add text stream to context / pending responses
            if isinstance(event.event, AgentTextStream):
                time.sleep(0.05)
                if event.event.delta is None or event.event.delta == "":
                    continue
                
                if no_text:
                    print("\n‼️ Agent starts to generate text:")
                print(event.event.delta, end='', flush=True)
                no_text = False
                continue
            
            if isinstance(event.event, StructuredOutput):
                print("\n‼️ Found structured output:")
                print(event.event.message)
                if hasattr(event.event.message, 'transfer'):
                    event.event.message.transfer()
            
            # add tool history to memory
            else:
                print("\n‼️ New Event:")
                print(event)
            
    print("\n=== Test tool call query ===")
    asyncio.run(run_one_turn(tool_call_query))

    print("\n\n=== Test name query ===")
    asyncio.run(run_one_turn(name_query))