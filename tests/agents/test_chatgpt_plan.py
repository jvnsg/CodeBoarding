import json

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from openai import OpenAI
from pydantic import BaseModel

from agents.chatgpt_plan import ChatGPTPlanChatModel, ChatGPTPlanError, _input, _result
from agents.llm_errors import is_quota_exhausted, is_transient_llm_error
from infra.chatgpt_auth import ChatGPTAccounts


def response(output, status="completed"):
    return {
        "id": "resp_1",
        "object": "response",
        "status": status,
        "model": "gpt-test",
        "output": output,
        "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
    }


def call(name, arguments, call_id="call_1"):
    return {
        "type": "function_call",
        "name": name,
        "namespace": "codeboarding",
        "call_id": call_id,
        "id": "fc_" + call_id,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def text(value):
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": value, "annotations": []}],
    }


@pytest.fixture
def transport(monkeypatch, tmp_path):
    requests: list[httpx.Request] = []
    replies: list = []
    monkeypatch.setattr(ChatGPTAccounts, "access_token", lambda self: "test-oauth-token")

    def handle(request):
        requests.append(request)
        reply = replies.pop(0)
        events = reply if isinstance(reply, list) else [{"type": "response.completed", "response": reply}]
        data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})

    def client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    monkeypatch.setattr("agents.chatgpt_plan.OpenAI", client)
    model = ChatGPTPlanChatModel(model_name="gpt-test", auth_file=str(tmp_path / "auth.json"))
    return model, requests, replies


def test_streaming_request_contract(transport):
    model, requests, replies = transport
    replies.append(response([text("Hello")]))
    answer = model.invoke([SystemMessage("Analyze the repository"), HumanMessage("hello")])
    assert answer.content == "Hello"
    assert answer.usage_metadata["total_tokens"] == 15
    payload = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://api.openai.com/v1/responses"
    assert requests[0].headers["authorization"] == "Bearer test-oauth-token"
    assert payload["store"] is False and payload["stream"] is True
    assert payload["input"][0]["role"] == "developer"
    assert "temperature" not in payload and "max_output_tokens" not in payload
    assert "previous_response_id" not in payload
    assert "test-oauth-token" not in model.model_dump_json()


def test_real_langchain_agent_runs_tools_and_returns_structured_output(transport):
    model, requests, replies = transport

    @tool
    def read_file(path: str) -> str:
        """Read repository source."""
        assert path == "example.py"
        return "def example(): return 1"

    class Analysis(BaseModel):
        summary: str

    replies.extend(
        [
            response([call("read_file", {"path": "example.py"})]),
            response([call("Analysis", {"summary": "One function"}, "call_2")]),
        ]
    )
    agent = create_agent(
        model=model, tools=[read_file], system_prompt="Analyze source.", response_format=ToolStrategy(Analysis)
    )
    result = agent.invoke({"messages": [HumanMessage("Analyze example.py")]})
    assert result["structured_response"].summary == "One function"
    first, second = [json.loads(r.content) for r in requests]
    assert first["tools"][0]["type"] == "namespace"
    assert {t["name"] for t in first["tools"][0]["tools"]} == {"read_file", "Analysis"}
    assert any(i.get("type") == "function_call_output" and "example" in i["output"] for i in second["input"])
    assert any(i.get("type") == "function_call" and i["namespace"] == "codeboarding" for i in second["input"])


def test_preserves_encrypted_reasoning_and_call_ids():
    reasoning = {"type": "reasoning", "id": "rs_1", "encrypted_content": "encrypted", "summary": []}
    output = [reasoning, call("read_file", {"path": "a.py"})]
    message = _result(response(output)).generations[0].message
    history = _input([HumanMessage("go"), message, ToolMessage("source", tool_call_id="call_1")])
    assert history[1:3] == output
    assert history[-1] == {"type": "function_call_output", "call_id": "call_1", "output": "source"}


@pytest.mark.parametrize("event", ["response.failed", "response.incomplete"])
def test_terminal_failure_is_not_success(transport, event):
    model, _, replies = transport
    failed = response([], "failed")
    failed["error"] = {"code": "subscription_sharing_usage_limit_exceeded", "message": "Limit"}
    replies.append([{"type": event, "response": failed}])
    with pytest.raises(ChatGPTPlanError) as error:
        model.invoke("hello")
    assert is_quota_exhausted(error.value)
    assert not is_transient_llm_error(error.value)


@pytest.mark.parametrize(
    "events", [[], [{"type": "error", "code": "subscription_sharing_usage_unavailable", "message": "Busy"}]]
)
def test_missing_completion_is_failure(transport, events):
    model, _, replies = transport
    replies.append(events)
    with pytest.raises(ChatGPTPlanError):
        model.invoke("hello")


def test_rejects_unsupported_options_before_network(transport):
    model, requests, _ = transport
    with pytest.raises(ValueError, match="temperature"):
        model.invoke("hi", temperature=0)
    with pytest.raises(ValueError, match="stop"):
        model.invoke("hi", stop=["end"])
    assert not requests


def test_named_tool_choice(transport):
    model, requests, replies = transport
    replies.append(response([text("ok")]))
    model.bind_tools(
        [{"name": "test", "description": "test", "parameters": {"type": "object"}}],
        tool_choice={"type": "function", "function": {"name": "test"}},
    ).invoke("go")
    assert json.loads(requests[0].content)["tool_choice"] == {
        "type": "function",
        "name": "test",
        "namespace": "codeboarding",
    }
