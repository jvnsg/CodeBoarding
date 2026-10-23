"""LangChain adapter for ChatGPT plan usage through the public Responses API."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import OpenAI

from infra.chatgpt_auth import ChatGPTAccounts, RESOURCE

NAMESPACE = "codeboarding"


class ChatGPTPlanError(RuntimeError):
    def __init__(self, code: str, request_id: str = ""):
        self.code = code
        self.status_code = {
            "subscription_sharing_usage_limit_exceeded": 429,
            "subscription_sharing_usage_unavailable": 503,
            "subscription_sharing_user_unavailable": 503,
            "subscription_sharing_user_not_eligible": 403,
            "chatpass_v2_scope_not_authorized": 403,
            "subscription_sharing_invalid_user": 401,
            "stream_ended_without_completion": 503,
        }.get(code, 400)
        super().__init__(
            f"ChatGPT plan request failed: {code} (request {request_id or 'unknown'}). "
            "Check ChatGPT Settings > Usage or run `codeboarding auth chatgpt login`."
        )


class ChatGPTPlanChatModel(BaseChatModel):
    """Use OAuth credentials exclusively; never fall back to API-key billing."""

    model_name: str = "gpt-6.1-sol"
    auth_file: str = ""
    timeout: float = 600

    @property
    def _llm_type(self) -> str:
        return "chatgpt_plan"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any):
        functions = []
        for tool in tools:
            converted = convert_to_openai_tool(tool)
            if converted.get("type") != "function":
                raise ValueError("ChatGPT plan supports CodeBoarding function tools only.")
            functions.append({"type": "function", "strict": False, **converted["function"]})
        bound = dict(kwargs)
        if functions:
            bound["tools"] = [
                {
                    "type": "namespace",
                    "name": NAMESPACE,
                    "description": "CodeBoarding repository analysis and structured results.",
                    "tools": functions,
                }
            ]
        if tool_choice is not None:
            if tool_choice in ("any", True):
                tool_choice = "required"
            elif isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
                tool_choice = {"type": "function", "name": tool_choice, "namespace": NAMESPACE}
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                tool_choice = {
                    "type": "function",
                    "name": tool_choice.get("name") or tool_choice["function"]["name"],
                    "namespace": NAMESPACE,
                }
            bound["tool_choice"] = tool_choice
        return self.bind(**bound)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            raise ValueError("ChatGPT plan Responses requests do not support stop sequences.")
        unsupported = set(kwargs) - {"tools", "tool_choice", "parallel_tool_calls", "timeout"}
        if unsupported:
            raise ValueError(f"Unsupported ChatGPT plan options: {', '.join(sorted(unsupported))}")
        # Path resolution stays local; tokens are not part of the serializable model.
        accounts = ChatGPTAccounts(path=Path(self.auth_file).expanduser() if self.auth_file else None)
        payload = {
            "model": self.model_name,
            "input": _input(messages),
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            **{k: v for k, v in kwargs.items() if k != "timeout"},
        }
        with OpenAI(
            api_key=accounts.access_token(),
            base_url=RESOURCE,
            max_retries=0,
            timeout=kwargs.get("timeout", self.timeout),
        ) as client:
            with client.responses.create(**payload) as events:
                for event in events:
                    if event.type == "response.completed":
                        response = event.response.model_dump(exclude_none=True)
                        if response.get("status") != "completed":
                            raise ChatGPTPlanError("response_not_completed", response.get("id", ""))
                        return _result(response)
                    if event.type in {"response.failed", "response.incomplete"}:
                        response = event.response.model_dump(exclude_none=True)
                        raise ChatGPTPlanError(
                            response.get("error", {}).get("code", event.type), response.get("id", "")
                        )
                    if event.type == "error":
                        raise ChatGPTPlanError(str(getattr(event, "code", "stream_error")))
        raise ChatGPTPlanError("stream_ended_without_completion")


def _input(messages: list[BaseMessage]) -> list[dict]:
    items: list[dict] = []
    for message in messages:
        if isinstance(message, SystemMessage) or message.type == "developer":
            items.append({"role": "developer", "content": message.content})
        elif isinstance(message, HumanMessage):
            items.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content if isinstance(message.content, str) else json.dumps(message.content),
                }
            )
        elif isinstance(message, AIMessage):
            previous = message.additional_kwargs.get("chatgpt_output")
            if previous is not None:
                items.extend(previous)
            else:
                if message.content:
                    items.append({"role": "assistant", "content": message.content})
                for call in message.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "name": call["name"],
                            "namespace": NAMESPACE,
                            "call_id": call["id"],
                            "arguments": json.dumps(call["args"]),
                        }
                    )
        else:
            raise ValueError(f"Unsupported ChatGPT message type: {message.type}")
    return items


def _result(response: dict) -> ChatResult:
    output = response.get("output", [])
    texts, calls = [], []
    for item in output:
        if item["type"] == "message":
            for content in item.get("content", []):
                if content["type"] == "output_text":
                    texts.append(content["text"])
                elif content["type"] == "refusal":
                    raise ChatGPTPlanError("model_refusal", response.get("id", ""))
        elif item["type"] == "function_call":
            if item.get("namespace", NAMESPACE) != NAMESPACE:
                raise ValueError("Response used an unknown tool namespace.")
            args = json.loads(item["arguments"])
            if not isinstance(args, dict):
                raise ValueError("Function call arguments must be a JSON object.")
            calls.append({"name": item["name"], "args": args, "id": item["call_id"], "type": "tool_call"})
        elif item["type"] != "reasoning":
            raise ValueError(f"Unsupported ChatGPT response output: {item['type']}")
    usage = response.get("usage") or {}
    message = AIMessage(
        content="".join(texts),
        tool_calls=calls,
        additional_kwargs={"chatgpt_output": output},
        response_metadata={
            "model_name": response.get("model", ""),
            "id": response.get("id", ""),
            "finish_reason": "tool_calls" if calls else "stop",
        },
        usage_metadata={
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    )
    return ChatResult(generations=[ChatGeneration(message=message)])
