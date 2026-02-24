# voice_agent_flow

`voice_agent_flow` is a lightweight framework for building **voice-oriented LLM agent flows** on top of `pydantic_ai`.

It focuses on:
- streaming text output for low-latency voice playback,
- tool call event mapping,
- structured output protocols for **agent handoff** and **hangup**,
- external memory/history control (important for interrupted voice turns).

## Key Concepts

- **AgentNode**: wraps one agent's model, instruction, tools, and structured output schema.
- **SingleAgentRunner**: runs one agent and emits normalized events.
- **MultiAgentRunner**: coordinates multiple `AgentNode`s and switches agents on handoff.
- **PydanticMessageAdaptor (`pmsg`)**: converts simple role/content dicts into `pydantic_ai` message history.
- **HangUpNode / DoHangUp**: standard pattern to end a call gracefully.

## Event Model

The runners emit `AgentResult` objects with events such as:

- `AgentTextStream` – incremental text deltas for TTS.
- `ToolCallsOutputStart` – tool call argument generation starts.
- `ToolCallsOutput` – tool call request emitted.
- `ToolCallResult` – tool result event.
- `AgentHandoff` – structured output triggers transfer to another agent.
- `HangupSignal` – structured output triggers call termination.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If your environment does not already have runtime dependencies, install:

```bash
pip install openai pydantic pydantic-ai python-dotenv nest_asyncio
```

## Environment Configuration

`voice_agent_flow.load_env.load_environment()` loads environment variables from:

- default: `~/.env`
- optional custom path: `load_environment("/path/to/.env")`

For Azure OpenAI with the OpenAI Python SDK, set typical variables like:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
OPENAI_API_VERSION=2024-XX-XX
```

## Quick Start (Single Agent)

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent

from voice_agent_flow.llms import create_pydantic_azure_openai
from voice_agent_flow.agents import SingleAgentRunner, pmsg
from voice_agent_flow.agents.events import AgentTextStream, ToolCallsOutput, ToolCallResult


class Person(BaseModel):
    name: str
    age: int


def weather_tool(location: str) -> str:
    """Search weather for a location."""
    return f"The weather in {location} is sunny."


model = create_pydantic_azure_openai(model_name="gpt-4o-mini")

agent = Agent(
    model,
    tools=[weather_tool],
    instructions="You are a helpful assistant.",
    output_type=str | Person,
)

runner = SingleAgentRunner(agent)

message_history = pmsg.to_history([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Beijing?"},
])


async def main():
    async for result in runner.run(message_history=message_history):
        if isinstance(result.event, AgentTextStream):
            print(result.event.delta, end="")
        elif isinstance(result.event, ToolCallsOutput):
            print("\n[TOOL_CALL]", result.event.message)
        elif isinstance(result.event, ToolCallResult):
            print("\n[TOOL_RESULT]", result.event.message)


asyncio.run(main())
```

## Quick Start (Multi-Agent Handoff)

Structured outputs can define `transfer()` to move control to another agent.

```python
from pydantic import BaseModel, Field

from voice_agent_flow.llms import create_pydantic_azure_openai
from voice_agent_flow.agents.agent_node import AgentNode, HangUpNode, DoHangUp
from voice_agent_flow.agents import MultiAgentRunner


class PoliceCallBasicInfo(BaseModel):
    case_location: str
    case_type: str
    description: str
    caller_name: str

    def transfer(self) -> str:
        return "safety_suggestion"


class SafetySuggestionProvided(BaseModel):
    suggestion_provided: bool = Field(...)

    def transfer(self) -> str:
        return "hangup"


model = create_pydantic_azure_openai("gpt-4o-mini")
instruction = "You are a police call center agent."

agents = {
    "police_call_basic_info": AgentNode(
        name="police_call_basic_info",
        model=model,
        instruction=instruction,
        task_cls=PoliceCallBasicInfo | DoHangUp,
    ),
    "safety_suggestion": AgentNode(
        name="safety_suggestion",
        model=model,
        instruction=instruction,
        task_cls=SafetySuggestionProvided,
    ),
    "hangup": HangUpNode(model=model),
}

runner = MultiAgentRunner(
    agents=agents,
    entry_agent_name="police_call_basic_info",
    ending_message="好的，我们这就派人过去处理您的情况，请您保持电话畅通！",
)
```

## Handoff / Hangup Protocol

- If a structured output model has a callable `transfer()` method, `SingleAgentRunner` emits `AgentHandoff`.
- `MultiAgentRunner` switches `current_agent` to the returned target.
- Special target `"end"` returns an ending text stream.
- If output is `DoHangUp`, runner emits `HangupSignal` so the voice layer can end the call.

## Message Adaptation (`pmsg`)

`pmsg` supports conversion from plain dictionaries to `pydantic_ai` messages:

- `system`
- `user`
- `assistant` (text or tool call)
- `tool` (tool return)

`pmsg.to_history(...)` also merges adjacent same-kind request/response messages to match expected model history format.

## Project Layout

```text
voice_agent_flow/
  agents/
        agent_node.py
        events.py
        message_adaptor.py
        single_agent_runner.py
        multi_agent_runner.py
  llms/
        openai_provider.py
        pydantic_provider.py
  load_env.py

streaming/
  basic_streaming.ipynb
  runs.ipynb

multi-agents.ipynb
```

## Notebooks

- `streaming/basic_streaming.ipynb`: pydantic_ai message/event exploration.
- `streaming/runs.ipynb`: single runner with text + structured outputs.
- `multi-agents.ipynb`: end-to-end multi-agent handoff + hangup flow.

## Current Package Info

- Name: `voice_agent_flow`
- Version: `0.0.1`
- Description: Build LLM agent flows for voice applications.

