"""Provider selection and Responses API contract; no paid/network calls."""
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

import test_assistant_routing as routing_tests
import ai_assistant as assistant


class ProviderTests(unittest.TestCase):
    setUp = routing_tests.AssistantRoutingTests.setUp
    question = "Walk me through the planning tradeoffs for Greek Yogurt."

    def ask(self, question=None, **kwargs):
        return assistant.answer_planning_question(question or self.question, self.context, "local", "model", **kwargs)

    def test_deterministic_and_unsupported_questions_never_read_key_or_call_provider(self):
        questions = [
            "Which product needs the most attention?", "Compare Whole Milk and Greek Yogurt.",
            "Which product has the worst forecast accuracy?", "How much Sour Cream should I have on hand?",
            "Why is Greek Yogurt's reorder point higher than Whole Milk's?",
            "What is Whole Milk's lead time?", "What should I do about Greek Yogurt?",
            "Which product has the largest service gap?", "Give me a quick inventory summary.",
            "What supplier should I use for Greek Yogurt?", "Will Whole Milk demand increase next Christmas?",
        ]
        with patch.object(assistant, "get_openai_api_key", return_value="test-only") as key, \
             patch.object(assistant, "request_openai_response") as cloud, \
             patch.object(assistant, "request_ollama_chat") as local:
            for question in questions:
                with self.subTest(question=question):
                    self.assertTrue(self.ask(question))
            key.assert_not_called()
            cloud.assert_not_called()
            local.assert_not_called()

    def test_key_selects_openai_for_only_grounded_fallback(self):
        with patch.object(assistant, "get_openai_api_key", return_value="test-only"), \
             patch.object(assistant, "request_openai_response", return_value="Grounded explanation") as cloud, \
             patch.object(assistant, "request_ollama_chat") as local:
            self.assertEqual(self.ask(ollama_available=False), "Grounded explanation")
            cloud.assert_called_once()
            payload = cloud.call_args.args[1]
            self.assertEqual(payload["question_category"], "grounded_explanation")
            self.assertEqual(set(payload["products_by_name"]), set(self.context["products_by_name"]))
            self.assertNotIn("excess_quantity", payload["products_by_name"]["Sour Cream"])
            self.assertNotIn("test-only", json.dumps(payload))
            local.assert_not_called()

    def test_no_key_uses_local_provider(self):
        with patch.object(assistant, "request_openai_response") as cloud, \
             patch.object(assistant, "request_ollama_chat", return_value="Local explanation") as local:
            self.assertEqual(self.ask(), "Local explanation")
            local.assert_called_once()
            cloud.assert_not_called()

    def test_unavailable_local_provider_returns_calculated_facts(self):
        with patch.object(assistant, "request_openai_response") as cloud, \
             patch.object(assistant, "request_ollama_chat", side_effect=URLError("offline")) as local:
            answer = self.ask("Explain the planning states of Sour Cream and Whole Milk.")
            self.assertIn("explanation service is unavailable", answer)
            self.assertIn("Run down 712 units", answer)
            self.assertIn("1,087", answer)
            cloud.assert_not_called()
            local.assert_called_once()

    def test_cloud_failures_and_empty_results_do_not_call_second_provider(self):
        for failure in [RuntimeError("private provider error test-only"), ""]:
            with self.subTest(failure=type(failure).__name__), \
                 patch.object(assistant, "get_openai_api_key", return_value="test-only"), \
                 patch.object(assistant, "request_openai_response") as cloud, \
                 patch.object(assistant, "request_ollama_chat") as local:
                if isinstance(failure, Exception):
                    cloud.side_effect = failure
                else:
                    cloud.return_value = failure
                answer = self.ask()
                self.assertIn("calculated planning facts", answer)
                self.assertNotIn("test-only", answer)
                self.assertNotIn("private provider error", answer)
                cloud.assert_called_once()
                local.assert_not_called()

    def test_real_sdk_responses_request_uses_mock_transport(self):
        import httpx
        from openai import OpenAI
        calls = []
        def respond(request):
            calls.append(request)
            return httpx.Response(200, json={
                "id": "resp_test", "object": "response", "created_at": 0,
                "status": "completed", "model": "gpt-5-nano",
                "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": "Grounded response", "annotations": []}]}],
            })
        client = OpenAI(api_key="test-only", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        payload = assistant.route_planning_question(self.question, self.context)["context"]
        with patch("openai.OpenAI", return_value=client) as constructor:
            self.assertEqual(assistant.request_openai_response("test-only", payload, self.question), "Grounded response")
        constructor.assert_called_once_with(api_key="test-only", max_retries=0, timeout=30.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].url.path, "/v1/responses")
        body = json.loads(calls[0].content)
        self.assertEqual(body["model"], "gpt-5-nano")
        self.assertEqual(body["max_output_tokens"], 1200)
        self.assertEqual(body["reasoning"], {"effort": "minimal"})
        self.assertFalse(body["store"])
        for field in ["tools", "temperature", "previous_response_id"]:
            self.assertNotIn(field, body)
        self.assertEqual(json.loads(body["input"])["analysis"], payload)
        self.assertIn("current analysis does not contain it", body["instructions"])
        self.assertIn("Never invent a numeric value", body["instructions"])
        self.assertNotIn("test-only", body["input"])
        for rule in [
            "not a confirmed stockout", "does not calculate an exact stockout date",
            "planning horizons, not exact stockout timing", "Do not describe replenishment as late",
            "Do not assume orders can be accelerated or expedited", "lead-time changes",
            "prioritize replenishment planning", "review replenishment options",
            "target inventory level", "inventory on hand", "Never expose internal Python identifiers",
            "For emails and executive summaries", "Do not mechanically list every metric",
            "Never invent suppliers, causes, forecasts, or external information",
        ]:
            self.assertIn(rule, body["instructions"])


    def test_open_ended_email_uses_existing_grounded_writing_path(self):
        question = "Draft a manager email about the current planning situation."
        with patch.object(assistant, "get_openai_api_key", return_value="test-only"), \
             patch.object(assistant, "request_openai_response", return_value="Manager-facing explanation") as cloud, \
             patch.object(assistant, "request_ollama_chat") as local:
            self.assertEqual(self.ask(question), "Manager-facing explanation")
            cloud.assert_called_once()
            self.assertEqual(cloud.call_args.args[2], question)
            self.assertEqual(cloud.call_args.args[1]["question_category"], "grounded_explanation")
            local.assert_not_called()

    def test_incomplete_cloud_response_is_not_presented(self):
        with patch("openai.OpenAI") as constructor:
            constructor.return_value.__enter__.return_value.responses.create.return_value = SimpleNamespace(
                status="incomplete", output_text="Partial recommendation")
            self.assertEqual(assistant.request_openai_response("test-only", {}, "Explain"), "")


class CredentialLookupTests(unittest.TestCase):
    def test_environment_precedes_streamlit_secrets(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": " test-only "}), patch.object(assistant.st, "secrets") as secrets:
            self.assertEqual(assistant.get_openai_api_key(), "test-only")
            secrets.get.assert_not_called()

    def test_streamlit_secret_used_when_environment_blank(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": " "}), patch.object(assistant.st, "secrets", {"OPENAI_API_KEY": "test-only"}):
            self.assertEqual(assistant.get_openai_api_key(), "test-only")

    def test_missing_secrets_file_is_not_an_error(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch.object(assistant.st, "secrets") as secrets:
            secrets.get.side_effect = assistant.st.errors.StreamlitSecretNotFoundError("no secrets")
            self.assertIsNone(assistant.get_openai_api_key())
