from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic_ai import ModelRequest, ModelResponse
from pydantic_ai.messages import (
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

PydanticMessage = ModelRequest | ModelResponse


def parse_timestamp(value: datetime | str | None) -> datetime:
    """Normalize timestamps into timezone-aware datetimes (UTC by default)."""
    
    if value is None:
        return datetime.now(timezone.utc)

    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    raise TypeError(f"Invalid timestamp type: {type(value).__name__}")


class PydanticMessageAdaptor:
    """Build pydantic_ai message objects from simple role/content dictionaries."""

    def user(
        self,
        content: str,
        timestamp: datetime | str | None = None,
    ) -> ModelRequest:
        ts = parse_timestamp(timestamp)
        return ModelRequest(parts=[UserPromptPart(content=content, timestamp=ts)])

    def assistant(
        self,
        content: str,
        timestamp: datetime | str | None = None,
    ) -> ModelResponse:
        ts = parse_timestamp(timestamp)
        return ModelResponse(parts=[TextPart(content=content)], timestamp=ts)

    def tool_call(
        self,
        tool_name: str,
        args: str,
        tool_call_id: str = "fake",
        timestamp: datetime | str | None = None,
    ) -> ModelResponse:
        ts = parse_timestamp(timestamp)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=tool_name,
                    args=args,
                    tool_call_id=tool_call_id,
                )
            ],
            timestamp=ts,
        )

    def tool_return(
        self,
        tool_name: str,
        content: str,
        tool_call_id: str = "fake",
        timestamp: datetime | str | None = None,
    ) -> ModelRequest:
        ts = parse_timestamp(timestamp)
        return ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=tool_name,
                    content=content,
                    tool_call_id=tool_call_id,
                    timestamp=ts,
                )
            ]
        )

    def system(
        self,
        content: str,
        timestamp: datetime | str | None = None,
    ) -> ModelRequest:
        ts = parse_timestamp(timestamp)
        return ModelRequest(parts=[SystemPromptPart(content=content, timestamp=ts)])

    def from_dict(self, msg: Mapping[str, Any]) -> PydanticMessage:
        """Convert one dict message to pydantic_ai message format."""
        role = msg.get("role")
        timestamp = parse_timestamp(msg.get("timestamp"))

        if role == "system":
            return self.system(content=self._required(msg, "content"), timestamp=timestamp)

        if role == "user":
            return self.user(content=self._required(msg, "content"), timestamp=timestamp)

        if role == "tool":
            return self.tool_return(
                tool_name=self._required(msg, "tool_name"),
                content=self._required(msg, "content"),
                tool_call_id=msg.get("tool_call_id", "fake"),
                timestamp=timestamp,
            )

        if role == "assistant":
            if "content" in msg:
                return self.assistant(content=msg["content"], timestamp=timestamp)

            if "tool_name" in msg:
                return self.tool_call(
                    tool_name=msg["tool_name"],
                    args=msg.get("args", ""),
                    tool_call_id=msg.get("tool_call_id", "fake"),
                    timestamp=timestamp,
                )

            raise ValueError(f"Invalid assistant message format: {msg}")

        raise ValueError(f"Unknown role: {role}")

    def to_history(self, 
                message_history: list[Mapping[str, Any]]) -> list[PydanticMessage]:
        """Convert then merge adjacent same-kind messages for pydantic_ai."""
        
        history = [self.from_dict(msg) for msg in message_history]

        merged_history: list[PydanticMessage] = []
        for msg in history:
            if not merged_history:
                merged_history.append(msg)
                continue

            last_msg = merged_history[-1]
            if isinstance(last_msg, ModelRequest) and isinstance(msg, ModelRequest):
                last_msg.parts.extend(msg.parts)
            elif isinstance(last_msg, ModelResponse) and isinstance(msg, ModelResponse):
                last_msg.parts.extend(msg.parts)
            else:
                merged_history.append(msg)

        return merged_history

    @staticmethod
    def _required(msg: Mapping[str, Any], key: str) -> Any:
        if key not in msg:
            raise ValueError(f"Missing required key '{key}' in message: {msg}")
        return msg[key]
    
pmsg = PydanticMessageAdaptor()