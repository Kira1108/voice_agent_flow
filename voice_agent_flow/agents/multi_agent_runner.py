from __future__ import annotations
from typing import Any, AsyncGenerator, Dict

from pydantic_ai import Agent

from voice_agent_flow.agents.events import (
    AgentResult, AgentTextStream, EventType, AgentHandoff, HangupSignal)
from voice_agent_flow.agents.single_agent_runner import SingleAgentRunner
from voice_agent_flow.agents.agent_node import AgentNode


class MultiAgentRunner:
    def __init__(
        self,
        agents: Dict[str, AgentNode],
        entry_agent_name: str,
        ending_message: str | None = None,
    ):
        # multi-agent container and cache
        self.agents = agents
        self._agent_cache: dict[str, Agent] = {}

        # entry agent and current agent
        self.entry_agent = self.get_agent(entry_agent_name)
        self.current_agent = self.entry_agent

        self.runner = SingleAgentRunner(agent=self.current_agent)

        self.agent_state: dict = {}
        self.ending_message = ending_message

    def get_agent(self, name: str) -> Agent:
        if name not in self._agent_cache:
            agent_node = self.agents[name]
            self._agent_cache[name] = agent_node.create()
        return self._agent_cache[name]
    
    def set_agent(self, name:str) -> None:
        if name not in self.agents:
            raise ValueError(f"Agent '{name}' not found in agents configuration.")
        
        agent = self.get_agent(name)
        self.current_agent = agent
        self.runner.set_agent(agent)

    async def _run(
        self,
        prompt: str | None = None,
        message_history: list | None = None,
    ) -> AsyncGenerator[AgentResult, None]:
        """
        Run only one turn.
        If StructuredOutput indicates handoff, switch agent and stop.
        Voice layer is responsible for rebuilding message_history and triggering next turn.
        """
        async for result in self.runner.run(
            prompt=prompt,
            message_history=message_history,
        ):
            if isinstance(result.event, AgentHandoff):
                yield self._handle_handoff(result)
                return
            
            if isinstance(result.event, HangupSignal):
                yield result
                return 
            
            yield result
            
    async def run(
        self, prompt: str | None = None, message_history: list | None = None
    ) -> AsyncGenerator[AgentResult, None]:
        """Run multiple turns until handoff or hangup."""
        rerun = False
        
        async for result in self._run(prompt=prompt, message_history=message_history):
            if isinstance(result.event, AgentHandoff):
                rerun = True
            yield result
            
        if rerun:
            async for result in self.run(message_history=message_history):
                yield result
        

    def _handle_handoff(self, result: AgentResult) -> AgentResult:
        """Handle handoff side effects and return the emitted event result for this turn."""
        output = result.event.message
        handoff_target = self._extract_handoff_target(output)

        if handoff_target is None:
            current_agent_name = getattr(self.current_agent, "name", "unknown")
            output_type = type(output).__name__
            raise RuntimeError(
                "AgentHandoff protocol violation: transfer() returned None "
                f"(current_agent='{current_agent_name}', output_type='{output_type}')."
            )

        if handoff_target == "end":
            return AgentResult(
                event=AgentTextStream(
                    delta=self.ending_message or "感谢您的接听，祝您生活愉快，再见！"
                ),
                event_type=EventType.AgentTextStream,
            )

        result.event.message = {
            "source_agent_name": self.current_agent.name,
            "target_agent_name": handoff_target,
        }

        self.current_agent = self.get_agent(handoff_target)
        self.runner.set_agent(self.current_agent)

        if hasattr(output, "model_dump"):
            self.agent_state.update(output.model_dump())

        return result
                
    def _extract_handoff_target(self, output: Any) -> str | None:
        """Return target agent name if output requests handoff; otherwise None."""
        transfer = getattr(output, "transfer", None)
        if not callable(transfer):
            return None

        target = transfer()
        
        if target not in self.agents and target != "end":
            raise ValueError(f"Handoff target '{target}' is not a valid agent or 'end'.")
        
        if isinstance(target, str) and target:
            return target

        return None










