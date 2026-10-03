import unittest
from unittest.mock import patch

from app.core.config import Settings
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, AiToolCall, _parse_openai_tool_decision
from app.services.tool_governance import ToolPolicyRegistry


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class AiToolCallingTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_request_and_response_use_native_tool_call_fields(self):
        captured = {}
        response_payload = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_next",
                                "type": "function",
                                "function": {
                                    "name": "mindbridge_case_create",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                }
            ]
        }

        class FakeAsyncClient:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def post(self, url, headers, json):
                captured.update(url=url, headers=headers, payload=json)
                return FakeResponse(response_payload)

        settings = Settings(
            _env_file=None,
            ai_provider="openai",
            openai_base_url="https://model.example/v1",
            openai_api_key="test-key",
            openai_model="test-model",
        )
        previous_call = AiToolCall(
            id="call_previous",
            name="mindbridge_excel_report",
            arguments="{}",
        )
        messages = [
            AiMessage(role="system", content="Choose the next required tool."),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [previous_call.assistant_tool_call()],
            },
            previous_call.result_message('{"status":"SUCCESS"}'),
        ]

        with patch("app.services.ai.httpx.AsyncClient", FakeAsyncClient):
            decision = await AiClient(settings).complete_with_tools(
                messages,
                ToolPolicyRegistry.function_tools(),
            )

        self.assertEqual(captured["url"], "https://model.example/v1/chat/completions")
        self.assertEqual(captured["payload"]["tool_choice"], "auto")
        self.assertIs(captured["payload"]["parallel_tool_calls"], False)
        self.assertIs(captured["payload"]["stream"], False)
        self.assertEqual(captured["payload"]["messages"][2]["role"], "tool")
        self.assertEqual(captured["payload"]["messages"][2]["tool_call_id"], "call_previous")
        self.assertEqual(len(captured["payload"]["tools"]), 3)
        self.assertEqual(decision.finish_reason, "tool_calls")
        self.assertEqual(decision.tool_calls[0].id, "call_next")
        self.assertEqual(decision.tool_calls[0].arguments_object(), {})

    async def test_mock_provider_returns_each_native_call_then_stops(self):
        settings = Settings(_env_file=None, ai_provider="mock")
        client = AiClient(settings)
        tools = ToolPolicyRegistry.function_tools()[:2]

        first = await client.complete_with_tools([], tools)
        first_call = first.tool_calls[0]
        messages = [
            first.assistant_message(),
            first_call.result_message('{"status":"SUCCESS"}'),
        ]
        second = await client.complete_with_tools(messages, tools)
        second_call = second.tool_calls[0]
        messages.extend(
            [
                second.assistant_message(),
                second_call.result_message('{"status":"SUCCESS"}'),
            ]
        )
        final = await client.complete_with_tools(messages, tools)

        self.assertEqual(first_call.name, "mindbridge_excel_report")
        self.assertEqual(second_call.name, "mindbridge_case_create")
        self.assertEqual(final.tool_calls, ())
        self.assertEqual(final.finish_reason, "stop")

    async def test_ollama_uses_object_arguments_and_tool_name(self):
        captured = {}
        response_payload = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "mindbridge_excel_report",
                            "arguments": {},
                        }
                    }
                ],
            },
            "done_reason": "stop",
        }

        class FakeAsyncClient:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def post(self, url, json):
                captured.update(url=url, payload=json)
                return FakeResponse(response_payload)

        settings = Settings(
            _env_file=None,
            ai_provider="ollama",
            ollama_base_url="http://ollama.example",
            ollama_model="test-model",
        )
        with patch("app.services.ai.httpx.AsyncClient", FakeAsyncClient):
            decision = await AiClient(settings).complete_with_tools(
                [AiMessage(role="system", content="Choose a tool.")],
                ToolPolicyRegistry.function_tools(),
            )

        self.assertEqual(captured["url"], "http://ollama.example/api/chat")
        self.assertNotIn("strict", captured["payload"]["tools"][0]["function"])
        self.assertEqual(decision.tool_calls[0].arguments_object(), {})
        self.assertNotIn("id", decision.assistant_message()["tool_calls"][0])
        self.assertEqual(
            decision.assistant_message()["tool_calls"][0]["function"]["arguments"],
            {},
        )
        self.assertEqual(
            decision.tool_calls[0].result_message('{"status":"SUCCESS"}'),
            {
                "role": "tool",
                "tool_name": "mindbridge_excel_report",
                "content": '{"status":"SUCCESS"}',
            },
        )

    async def test_unknown_provider_is_rejected_explicitly(self):
        settings = Settings(_env_file=None, ai_provider="custom")

        with self.assertRaisesRegex(RuntimeError, "AI_PROVIDER=custom"):
            await AiClient(settings).complete_with_tools([], ToolPolicyRegistry.function_tools())

    def test_decision_builds_assistant_and_correlated_tool_messages(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "mindbridge_excel_report",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                }
            ]
        }

        decision = _parse_openai_tool_decision(payload)
        assistant_message = decision.assistant_message()
        tool_message = decision.tool_calls[0].result_message('{"status":"SUCCESS"}')

        self.assertEqual(assistant_message["tool_calls"][0]["id"], "call_123")
        self.assertEqual(tool_message["tool_call_id"], "call_123")
        self.assertEqual(tool_message["role"], "tool")

    def test_arguments_must_be_a_json_object(self):
        with self.assertRaisesRegex(ValueError, "合法 JSON"):
            AiToolCall(id="call_bad", name="bad", arguments="not-json").arguments_object()
        with self.assertRaisesRegex(ValueError, "JSON object"):
            AiToolCall(id="call_list", name="bad", arguments="[]").arguments_object()

    def test_malformed_provider_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "缺少 choices"):
            _parse_openai_tool_decision({})


if __name__ == "__main__":
    unittest.main()
