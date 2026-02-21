from __future__ import annotations
import logging
from typing import Any, AsyncGenerator, Dict

from pydantic_ai import Agent

from voice_agent_flow.agents.events import AgentResult, StructuredOutput
from voice_agent_flow.agents.single_agent_runner import SingleAgentRunner
from voice_agent_flow.node import AgentNode


class MultiAgentRunner:
    def __init__(
        self,
        agents: Dict[str, AgentNode],
        entry_agent_name: str,
        ending_message: str | None = None,
        max_iter: int = 3,
    ):
        # multi-agent container and cache
        self.agents = agents
        self._agent_cache: dict[str, Agent] = {}

        # entry agent and current agent
        self.entry_agent = self.get_agent(entry_agent_name)
        self.current_agent = self.entry_agent

        self.runner = SingleAgentRunner(agent=self.current_agent)

        self.agent_state: dict = {}
        self.max_iter = max_iter
        self.ending_message = ending_message

    def get_agent(self, name: str) -> Agent:
        if name not in self._agent_cache:
            agent_node = self.agents[name]
            self._agent_cache[name] = agent_node.create()
        return self._agent_cache[name]
    
    async def run(
        self,
        prompt: str | None = None,
        message_history: list | None = None,
    ) -> AsyncGenerator[AgentResult, None]:
        async for event in self._run_recursive(
            prompt=prompt,
            message_history=message_history,
            iter_idx=0,
        ):
            yield event

    async def _run_recursive(
        self,
        prompt: str | None = None,
        message_history: list | None = None,
        iter_idx: int = 0,
    ) -> AsyncGenerator[AgentResult, None]:
        """Run one agent turn and recursively continue when handoff is required."""

        if iter_idx >= self.max_iter:
            logging.info(f"Reached max iteration {self.max_iter}. Ending run.")
            return

        handoff_target: str | None = None

        async for event in self.runner.run(
            prompt=prompt,
            message_history=message_history,
        ):
            if not isinstance(event, StructuredOutput):
                yield event
                continue

            output = event.message
            handoff_target = self._extract_handoff_target(output)

            # No handoff requested, current run is complete.
            if handoff_target is None:
                yield event
                return

            # Handoff requested: switch current agent and continue recursively.
            agent = self.get_agent(handoff_target)
            self.current_agent = agent
            self.runner.set_agent(agent)

            if hasattr(output, "model_dump"):
                self.agent_state.update(output.model_dump())

            break

        if handoff_target is not None:
            async for event in self._run_recursive(
                prompt=prompt,
                message_history=message_history,
                iter_idx=iter_idx + 1,
            ):
                yield event

    def _extract_handoff_target(self, output: Any) -> str | None:
        """Return target agent name if output requests handoff; otherwise None."""
        transfer = getattr(output, "transfer", None)
        if not callable(transfer):
            return None

        target = transfer()
        if isinstance(target, str) and target:
            return target

        return None
    
    

        
    
    
    
    
    
            