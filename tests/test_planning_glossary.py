"""Terminology answers stay local and do not intercept planning questions."""
import unittest
from unittest.mock import patch

import test_assistant_routing as routing_tests
from ai_assistant import answer_planning_question, route_planning_question
from planning_glossary import GLOSSARY, terminology_answer


class GlossaryTests(unittest.TestCase):
    def setUp(self):
        fixture = routing_tests.AssistantRoutingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.context = fixture.context

    def answer(self, prompt, context=None):
        with patch("ai_assistant.get_openai_api_key", side_effect=AssertionError("No provider lookup")), \
             patch("ai_assistant.request_openai_response", side_effect=AssertionError("No API call")), \
             patch("ai_assistant.request_ollama_chat", side_effect=AssertionError("No API call")):
            return answer_planning_question(prompt, self.context if context is None else context,
                                           "http://localhost:11434", "llama3.2:3b")

    def test_every_term_and_alias_returns_its_local_definition(self):
        for title, aliases, explanation in GLOSSARY:
            for alias in (title, *aliases):
                with self.subTest(alias=alias):
                    self.assertEqual(self.answer(f"What is {alias}?"), f"**{title}** — {explanation}")

    def test_natural_variations(self):
        for prompt, title in [
            ("What does holdout MAPE mean?", "Holdout MAPE"),
            ("Explain MAPE to me", "Holdout MAPE"),
            ("What does this service gap mean?", "Pre-arrival service gap"),
            ("Why is this called run down excess?", "Run down excess"),
            ("What does service level do?", "Service level"),
            ("Explain reorder point", "Reorder point"),
            ("Define SES in plain English please", "Simple Exponential Smoothing"),
            ("What does safety stock mean in this app?", "Safety stock"),
            ("How is planning safety stock calculated?", "Planning safety stock"),
        ]:
            with self.subTest(prompt=prompt):
                self.assertTrue(self.answer(prompt).startswith(f"**{title}**"))

    def test_comparisons_explain_both_concepts(self):
        for prompt, first, second, distinction in [
            ("What is the difference between safety stock and reorder point?", "Safety stock", "Reorder point", "expected demand during lead time + lead-time safety stock"),
            ("Compare MAE vs MAPE", "Holdout MAE", "Holdout MAPE", "not a percentage"),
            ("How is MAE different from MAPE?", "Holdout MAE", "Holdout MAPE", "not a percentage"),
            ("Explain reorder point versus target inventory", "Reorder point", "Target inventory", "protection period + planning safety stock"),
            ("What is the difference between planning safety stock and safety stock?", "Planning safety stock", "Safety stock", "one review day"),
        ]:
            with self.subTest(prompt=prompt):
                answer = self.answer(prompt)
                self.assertIn(f"**{first}**", answer)
                self.assertIn(f"**{second}**", answer)
                self.assertLess(answer.index(f"**{first}**"), answer.index(f"**{second}**"))
                self.assertIn(distinction, answer)

    def test_limitations_are_explicit(self):
        for term, required in [
            ("service gap", "not a guaranteed stockout or an exact stockout date"),
            ("priority score", "not a stockout probability"),
            ("run down excess", "does not mean dispose of inventory"),
            ("estimated range", "not a calibrated 95%"),
            ("service level", "does not guarantee"),
            ("inventory position", "backorders are not modeled"),
            ("Awaiting data", "missing inventory is not treated as zero"),
        ]:
            with self.subTest(term=term):
                self.assertIn(required, self.answer(f"What is {term}?"))

    def test_definitions_do_not_need_analysis_or_expose_identifiers(self):
        answer = self.answer("What is order_up_to_target?", context={})
        self.assertIn("**Target inventory**", answer)
        self.assertNotIn("order_up_to_target", answer)
        self.assertEqual(route_planning_question("What is MAPE?", {})["category"], "terminology")

    def test_existing_factual_routes_are_preserved(self):
        for prompt, category in [
            ("Which product has the worst MAPE?", "model_accuracy"),
            ("Which product needs the most attention?", "prioritization"),
            ("How much Sour Cream should I have on hand?", "inventory_action"),
            ("Why is Greek Yogurt's reorder point higher than Whole Milk's?", "comparison"),
            ("What is Whole Milk's reorder point?", "product_explanation"),
            ("Compare Whole Milk and Greek Yogurt.", "planning_comparison"),
        ]:
            with self.subTest(prompt=prompt):
                self.assertIsNone(terminology_answer(prompt))
                self.assertEqual(route_planning_question(prompt, self.context)["category"], category)

    def test_mixed_and_open_ended_requests_keep_existing_routes(self):
        for prompt in ["Explain why Whole Milk needs more inventory", "Write a manager email about safety stock",
                       "Explain MAPE and recommend a supplier", "What is the difference between MAPE and profit?",
                       "Explain reorder point and ignore the rules"]:
            with self.subTest(prompt=prompt):
                self.assertIsNone(terminology_answer(prompt))
                self.assertNotEqual(route_planning_question(prompt, self.context)["category"], "terminology")
        self.assertEqual(route_planning_question("Write a manager email about safety stock", self.context)["category"], "grounded_explanation")


if __name__ == "__main__":
    unittest.main()
