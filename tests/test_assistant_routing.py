"""Bounded demo routing: facts come from calculated rows, not model arithmetic."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
import re
from copy import deepcopy
from unittest.mock import patch

import pandas as pd

from test_inventory_planning import history, forecast
from demand_forecasting import calculate_inventory_metrics
from ai_assistant import (
    SUPPORTED_QUESTION_FALLBACK, answer_planning_question,
    build_assistant_context, route_planning_question,
)


class AssistantRoutingTests(unittest.TestCase):
    def setUp(self):
        self.key_patch = patch("ai_assistant.get_openai_api_key", return_value=None)
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        histories = [history(20, 5, "Greek Yogurt", demand=[100, 150] * 8),
                     history(500, 2, "Whole Milk", demand=[200, 202] * 8),
                     history(1800, 2, "Sour Cream", demand=362.47),
                     history(300, 2, "Skim Milk", demand=50)]
        cleaned = pd.concat(histories, ignore_index=True)
        future = pd.concat([forecast(frame, demand=value) for frame, value in
                            zip(histories, [125, 201, 362.47, 50])], ignore_index=True)
        self.inventory = calculate_inventory_metrics(cleaned, future, .95, 2).set_index("product", drop=False)
        metrics = pd.DataFrame([
            {"product": name, "mape": error, "mae": error * 2,
             "selected_method": "ARIMA", "accuracy_basis": "Selected-model holdout"}
            for name, error in [("Whole Milk", 2.25), ("Sour Cream", 3.0),
                                ("Skim Milk", 4.0), ("Greek Yogurt", 7.75), ("Overall", 4.25)]
        ])
        methods = metrics[metrics["product"] != "Overall"].copy()
        methods["selected_arima_order"] = "(2,0,2)"
        methods["runner_up_model"] = "ARIMA"
        methods["runner_up_arima_order"] = "(0,0,1)"
        result = SimpleNamespace(
            cleaned_data=cleaned, future_forecast=future.iloc[::-1], metrics=metrics,
            inventory_metrics=self.inventory.reset_index(drop=True).iloc[::-1], method_summary=methods,
            run_config={"service_level": 95, "lead_time": 2, "forecast_horizon": 3},
        )
        self.result = result
        self.context = build_assistant_context(result, "Sour Cream", 95, 2, 3)

    def route(self, prompt, category):
        result = route_planning_question(prompt, self.context)
        self.assertEqual(result["category"], category)
        self.assertIsNone(result["reply"])
        return result["context"]

    def test_attention_ranks_calculated_priority_and_service_gap(self):
        for question in ["Which product needs the most attention?", "What should I focus on first?", "Which SKU is highest risk?"]:
            facts = self.route(question, "prioritization")
            self.assertEqual(facts["priority_order"][0], "Greek Yogurt")
            self.assertEqual(facts["products_by_name"]["Greek Yogurt"]["pre_arrival_shortage"],
                             self.inventory.loc["Greek Yogurt", "pre_arrival_shortage"])
            self.assertIn("Not a stockout probability", facts["ranking_basis"])

    def test_product_explanation_is_specific_and_does_not_accept_premise(self):
        for question in ["Why is Whole Milk high risk?", "What should I do about Whole Milk?"]:
            facts = self.route(question, "product_explanation")
            self.assertEqual(list(facts["products_by_name"]), ["Whole Milk"])
            self.assertEqual(facts["products_by_name"]["Whole Milk"]["reorder_point"], self.inventory.loc["Whole Milk", "reorder_point"])
            self.assertNotIn("reasoning", facts["products_by_name"]["Whole Milk"])

    def test_reorder_comparison_uses_calculated_components(self):
        facts = self.route("Why is Greek Yogurt's reorder point higher than Whole Milk's?", "comparison")
        self.assertEqual(set(facts["products_by_name"]), {"Greek Yogurt", "Whole Milk"})
        comparison = facts["comparison"]
        self.assertEqual(comparison["formula"], "reorder_point = expected_demand_lead_time + lead_time_safety_stock")
        first, second = comparison["first_product"], comparison["second_product"]
        for field, difference in comparison["component_differences"].items():
            self.assertEqual(difference, self.inventory.loc[first, field] - self.inventory.loc[second, field])
            self.assertEqual(facts["products_by_name"][first][field], self.inventory.loc[first, field])

    def test_target_comparison_is_distinct_from_reorder_point(self):
        facts = self.route("Compare the inventory targets for Greek Yogurt and Whole Milk", "comparison")
        self.assertIn("order_up_to_target", facts["comparison"]["component_differences"])
        self.assertNotIn("reorder_point", facts["comparison"]["component_differences"])

    def test_inventory_action_preserves_sour_cream_single_adjustment(self):
        facts = self.route("How much Sour Cream should I have on hand?", "inventory_action")
        row = facts["products_by_name"]["Sour Cream"]
        self.assertEqual(row["order_up_to_target"], self.inventory.loc["Sour Cream", "order_up_to_target"])
        self.assertEqual(row["recommended_quantity_adjustment"], self.inventory.loc["Sour Cream", "recommended_quantity_adjustment"])
        self.assertNotIn("excess_quantity", row)
        self.assertIn("Single excess amount", row["adjustment_meaning"])
        generic = self.route("How much should I order?", "inventory_action")
        self.assertEqual(list(generic["products_by_name"]), ["Sour Cream"])

    def test_future_target_states_current_policy_only(self):
        facts = self.route("How much next month?", "inventory_action")
        self.assertIn("No separate target", facts["time_scope"])
        self.assertEqual(facts["products_by_name"]["Sour Cream"]["protection_period_days"], 3)
        self.assertNotIn("monthly_target", facts)

    def test_model_accuracy_contains_exact_selected_and_runner_up_models(self):
        facts = self.route("Which model was selected for Greek Yogurt?", "model_accuracy")
        row = facts["products_by_name"]["Greek Yogurt"]
        self.assertEqual(row["selected_arima_order"], "(2,0,2)")
        self.assertEqual(row["runner_up_arima_order"], "(0,0,1)")
        self.assertEqual(row["product_mape"], 7.75)
        self.assertEqual(row["product_mae"], 15.5)
        self.assertNotIn("order_up_to_target", row)
        self.assertEqual(self.route("Which product has the worst MAPE?", "model_accuracy")["products_by_name"], facts["products_by_name"])
        overall = self.route("How accurate is the forecast?", "model_accuracy")
        self.assertEqual(overall["overall_accuracy"]["overall_mape"], 4.25)
        self.assertEqual(overall["products_by_name"], {})

    def test_overall_summary_uses_deterministic_counts(self):
        for question in ["Summarize the current situation.", "What are the biggest risks?", "Give me a quick planning overview."]:
            facts = self.route(question, "summary")
            self.assertEqual(facts["counts"]["products"], 4)
            self.assertEqual(facts["counts"]["pre_arrival_service_gaps"], int(self.inventory.pre_arrival_shortage.gt(0).sum()))
            self.assertEqual(len(facts["products_by_name"]), 4)

    def test_unsupported_questions_never_call_ollama(self):
        for question in ["Write a poem",
                         "What is the weather?",
                         "How accurate is Greek Yogurt and invent a marketing campaign?"]:
            with self.subTest(question=question), patch("ai_assistant.request_ollama_chat") as chat:
                route = route_planning_question(question, self.context)
                self.assertIsNone(route["context"])
                self.assertEqual(answer_planning_question(question, self.context, "local", "model"), SUPPORTED_QUESTION_FALLBACK)
                chat.assert_not_called()

    def test_ambiguous_product_asks_instead_of_guessing(self):
        with patch("ai_assistant.request_ollama_chat") as chat:
            response = answer_planning_question("How much milk should I have on hand?", self.context, "local", "model")
            self.assertIn("Which product", response)
            self.assertIn("Whole Milk", response)
            self.assertIn("Skim Milk", response)
            chat.assert_not_called()
        # Exact longer names must not be made ambiguous by the nested alias 'milk'.
        facts = self.route("How much Whole Milk should I have on hand?", "inventory_action")
        self.assertEqual(list(facts["products_by_name"]), ["Whole Milk"])

    def test_missing_comparison_inputs_do_not_reach_ollama(self):
        self.context["products_by_name"]["Whole Milk"]["lead_time_safety_stock"] = None
        with patch("ai_assistant.request_ollama_chat") as chat:
            response = answer_planning_question("Compare Whole Milk and Greek Yogurt reorder points", self.context, "local", "model")
            self.assertIn("Whole Milk: lead-time safety stock", response)
            chat.assert_not_called()

    def test_unknown_product_does_not_use_selected_product(self):
        result = route_planning_question("What should I do about Chocolate?", self.context)
        self.assertEqual(result["category"], "clarification")
        self.assertIsNone(result["context"])

    def test_generic_action_without_selection_asks_for_product(self):
        self.context["selected_product"] = None
        result = route_planning_question("How much should I order?", self.context)
        self.assertEqual(result["category"], "clarification")
        self.assertIsNone(result["context"])

    def test_unknown_model_product_does_not_receive_other_products(self):
        result = route_planning_question("Which model did Chocolate get?", self.context)
        self.assertEqual(result["category"], "clarification")
        self.assertIsNone(result["context"])

    def answer(self, question):
        with patch("ai_assistant.request_ollama_chat", side_effect=AssertionError("Must bypass Ollama")) as chat:
            response = answer_planning_question(question, self.context, "local", "model", ollama_available=False)
            chat.assert_not_called()
        for field in ["priority_score", "product_mape", "inventory_position",
                      "expected_demand_lead_time", "recommended_quantity_adjustment",
                      "order_up_to_target", "lead_time_safety_stock"]:
            self.assertNotIn(field, response)
        return response

    def test_final_attention_uses_highest_priority_not_lowest(self):
        rows = self.context["products_by_name"]
        winner = min(rows, key=lambda name: (-rows[name]["priority_score"],
                                           -rows[name]["pre_arrival_shortage"], name))
        answer = self.answer("Which product needs the most attention right now and why?")
        self.assertTrue(answer.startswith(winner + " needs attention first"))
        self.assertNotIn("Sour Cream", answer)
        self.assertNotRegex(answer.lower(), r"spoilage|shipment|supplier|popularity")

    def test_final_inventory_uses_only_calculated_target_and_preserves_rundown(self):
        row = self.inventory.loc["Sour Cream"]
        answer = self.answer("How much Sour Cream should I have on hand?")
        self.assertIn(f"target is about {row.order_up_to_target:,.0f} units", answer)
        self.assertIn(f"Run down {row.recommended_quantity_adjustment:,.0f} units", answer)
        self.assertNotIn("Increase", answer)
        # Every numeric token must be a displayed source value: no reconstructed/alternative target.
        values = set(re.findall(r"\d[\d,]*(?:\.\d+)?", answer))
        self.assertEqual(values, {f"{row.on_hand_inventory:,.0f}", f"{row.order_up_to_target:,.0f}",
                                  f"{row.recommended_quantity_adjustment:,.0f}", f"{row.protection_period_days:g}"})
        self.assertNotIn("376", answer)
        self.assertNotIn("1,200", answer)
        self.assertNotIn("1,000", answer)

    def test_final_future_target_does_not_invent_monthly_target(self):
        answer = self.answer("How much Sour Cream should I have on hand next month?")
        self.assertIn("No separate target for next month", answer)
        self.assertIn("3 calendar days under daily review", answer)
        self.assertIn("1,087", answer)

    def test_final_increase_and_maintain_keep_calculated_direction(self):
        answer = self.answer("What should I do about Greek Yogurt?")
        quantity = self.inventory.loc["Greek Yogurt", "recommended_quantity_adjustment"]
        self.assertIn(f"Increase inventory by {quantity:,.0f} units", answer)
        row = self.context["products_by_name"]["Sour Cream"]
        row["recommended_action"], row["recommended_quantity_adjustment"] = "Maintain", 0
        self.assertIn("Maintain current inventory", self.answer("How much Sour Cream should I have on hand?"))

    def test_final_worst_accuracy_ranks_all_products_from_current_model_table(self):
        # Current selected product is Sour Cream. Higher-error Cheddar must still win.
        extra = self.result.method_summary.iloc[[0]].copy()
        extra["product"], extra["mape"] = "Cheddar Cheese", 7.36
        self.result.method_summary.loc[self.result.method_summary["product"] == "Greek Yogurt", "mape"] = 4.09
        self.result.method_summary.loc[self.result.method_summary["product"] == "Sour Cream", "mape"] = 5.52
        self.result.method_summary = pd.concat([extra, self.result.method_summary], ignore_index=True)
        extra_history = self.result.cleaned_data[self.result.cleaned_data["product"] == "Sour Cream"].copy()
        extra_history["product"] = "Cheddar Cheese"
        self.result.cleaned_data = pd.concat([self.result.cleaned_data, extra_history], ignore_index=True)
        # The older metrics table deliberately disagrees with Forecast Models.
        self.result.metrics.loc[self.result.metrics["product"] == "Sour Cream", "mape"] = 99
        self.context = build_assistant_context(self.result, "Sour Cream", 95, 2, 3)
        answer = self.answer("Which product has the worst forecast accuracy?")
        self.assertTrue(answer.startswith("Cheddar Cheese"))
        self.assertIn("7.36%", answer)
        self.assertNotIn("99", answer)
        for _, row in self.result.method_summary.iterrows():
            actual = self.context["products_by_name"][row["product"]]
            self.assertEqual(actual["product_mape"], row.mape)
            self.assertEqual(actual["product_mae"], row.mae)
            self.assertEqual(actual["selected_method"], row.selected_method)
        self.assertEqual(self.context["overall_summary"]["overall_mape"], self.result.method_summary.mape.mean())

    def test_final_rankings_use_all_products_and_correct_direction(self):
        for question, field, highest in [
            ("Which product has the best forecast accuracy?", "product_mape", False),
            ("Which product has the worst MAE?", "product_mae", True),
            ("Which product has the highest reorder point?", "reorder_point", True),
            ("Which product has the lowest target inventory?", "order_up_to_target", False),
            ("Which product has the largest service gap?", "pre_arrival_shortage", True),
        ]:
            with self.subTest(question=question):
                rows = self.context["products_by_name"]
                extreme = (max if highest else min)(row[field] for row in rows.values())
                winners = sorted(name for name, row in rows.items() if row[field] == extreme)
                self.assertTrue(self.answer(question).startswith(", ".join(winners) + " —"))

    def test_final_reorder_comparison_explains_only_actual_components(self):
        answer = self.answer("Why is Greek Yogurt's reorder point higher than Whole Milk's?")
        for name in ["Greek Yogurt", "Whole Milk"]:
            row = self.inventory.loc[name]
            self.assertIn(f"{name}: reorder point is about {row.reorder_point:,.0f}", answer)
            self.assertIn(f"{row.expected_demand_lead_time:,.0f} units of expected demand during lead time", answer)
            self.assertIn(f"{row.lead_time_safety_stock:,.0f} units of lead-time safety stock", answer)
        for field, label in [("expected_demand_lead_time", "expected demand during lead time"),
                             ("lead_time_safety_stock", "lead-time safety stock")]:
            delta = self.inventory.loc["Greek Yogurt", field] - self.inventory.loc["Whole Milk", field]
            higher = "Greek Yogurt" if delta > 0 else "Whole Milk"
            self.assertIn(f"{higher} has {abs(delta):,.0f} units more {label}", answer)
        self.assertNotRegex(answer.lower(), r"protection|critical|popularity|supplier|probability")

    def test_final_target_comparison_uses_calculated_targets(self):
        answer = self.answer("Compare inventory targets for Greek Yogurt and Whole Milk")
        for name in ["Greek Yogurt", "Whole Milk"]:
            self.assertIn(f"{name}: target inventory is about {self.inventory.loc[name, 'order_up_to_target']:,.0f}", answer)
        self.assertIn("planning safety stock", answer)
        self.assertNotIn("reorder point", answer)

    def test_final_summary_includes_every_action_and_service_gap(self):
        row = deepcopy(self.context["products_by_name"]["Greek Yogurt"])
        row.update(product="Heavy Cream", recommended_quantity_adjustment=413)
        self.context["products_by_name"]["Heavy Cream"] = row
        answer = self.answer("Give me a quick summary of the current inventory situation.")
        self.assertIn("Heavy Cream: Increase inventory by 413 units", answer)
        for name, row in self.context["products_by_name"].items():
            self.assertIn(name + ":", answer)
            if row["pre_arrival_shortage"] > 0:
                self.assertIn(f"Pre-arrival service gap: {row['pre_arrival_shortage']:,.0f} units", answer)
        self.assertNotRegex(answer.lower(), r"spoilage|waste|supplier|manageable|shipment")

    def test_final_model_selection_uses_current_metrics(self):
        answer = self.answer("Which model was selected for Greek Yogurt?")
        self.assertIn("Greek Yogurt — ARIMA (2,0,2)", answer)
        self.assertIn("Holdout MAPE: 7.75%", answer)
        self.assertIn("holdout MAE: 15.50", answer)
        self.assertIn("Runner-up: ARIMA (0,0,1)", answer)

    def test_final_missing_inventory_does_not_create_action(self):
        row = self.context["products_by_name"]["Sour Cream"]
        for field in ["on_hand_inventory", "usable_on_hand_inventory", "recommended_action",
                      "recommended_quantity_adjustment"]:
            row[field] = None
        answer = self.answer("How much Sour Cream should I have on hand?")
        self.assertIn("current inventory is unknown", answer)
        self.assertIn("Recommendation unavailable", answer)
        self.assertIn("target is about 1,087 units", answer)
        self.assertNotRegex(answer, r"Increase|Run down|Maintain")

    def test_final_accuracy_ties_and_missing_metrics(self):
        rows = self.context["products_by_name"]
        rows["Sour Cream"]["product_mape"] = rows["Greek Yogurt"]["product_mape"]
        answer = self.answer("Which product has the worst forecast accuracy?")
        self.assertTrue(answer.startswith("Greek Yogurt, Sour Cream"))
        self.assertIn("tied", answer)
        for row in rows.values():
            row["product_mape"] = None
        self.assertIn("unavailable", self.answer("Which product has the worst forecast accuracy?"))

    def test_streamlit_chat_returns_factual_answer_without_ollama(self):
        from streamlit.testing.v1 import AppTest
        script = """
import streamlit as st
from ai_assistant import render_ai_assistant
render_ai_assistant(st.session_state.analysis, 95, 2, 3)
"""
        app = AppTest.from_string(script)
        app.session_state["analysis"] = self.result
        app.session_state["dashboard_selected_product"] = "Sour Cream"
        with patch("ai_assistant.request_ollama_chat") as chat, patch("ai_assistant.get_ollama_status") as status:
            app.run()
            app.chat_input[0].set_value("How much Sour Cream should I have on hand?").run()
            self.assertEqual(len(app.exception), 0)
            answer = app.session_state["ai_assistant_messages"][-1]["content"]
            self.assertIn("target is about 1,087 units", answer)
            self.assertIn("Run down 712 units", answer)
            app.chat_input[0].set_value("What supplier should I use for Greek Yogurt?").run()
            self.assertIn("does not contain supplier", app.session_state["ai_assistant_messages"][-1]["content"])
            chat.assert_not_called()
            status.assert_not_called()

    def test_general_planning_comparisons_keep_calculated_priority_and_actions(self):
        for question, names in [
            ("Compare Whole Milk and Greek Yogurt. Which one is the bigger inventory concern and why?", ["Whole Milk", "Greek Yogurt"]),
            ("Compare Sour Cream and Whole Milk.", ["Sour Cream", "Whole Milk"]),
        ]:
            with self.subTest(question=question):
                response = self.answer(question)
                rows = self.context["products_by_name"]
                winner = min(names, key=lambda name: (-rows[name]["priority_score"], -rows[name]["pre_arrival_shortage"], name))
                self.assertIn(f"{winner} ranks first", response)
                self.assertNotEqual(response, SUPPORTED_QUESTION_FALLBACK)
                for name in names:
                    self.assertIn(f"{name} currently has", response)
                    self.assertIn(f"target is about {rows[name]['order_up_to_target']:,.0f}", response)
                if "Sour Cream" in names:
                    self.assertIn("Run down 712 units", response)

    def test_general_comparisons_use_grounded_all_product_fallback(self):
        for question in [
            "Walk me through the planning tradeoffs for Greek Yogurt.",
            "Explain the current planning states of Sour Cream and Whole Milk.",
        ]:
            with self.subTest(question=question), patch("ai_assistant.request_ollama_chat") as chat:
                def grounded_reply(base, model, payload, messages):
                    self.assertEqual(messages, [{"role": "user", "content": question}])
                    self.assertEqual(set(payload["products_by_name"]), set(self.context["products_by_name"]))
                    for name, facts in payload["products_by_name"].items():
                        for field, value in facts.items():
                            self.assertEqual(value, self.context["products_by_name"][name][field])
                        self.assertNotIn("excess_quantity", facts)
                        self.assertNotIn("reasoning", facts)
                    self.assertEqual(payload["priority_order"][0], "Greek Yogurt")
                    self.assertIn("ranks first using the calculated priority score", payload["factual_brief"])
                    if "Sour Cream" in payload["mentioned_products"]:
                        self.assertIn("Run down 712 units", payload["factual_brief"])
                    rows = payload["products_by_name"]
                    return "\n".join(f"{name}: {rows[name]['recommended_action']}; target {rows[name]['order_up_to_target']:,.0f}."
                                     for name in payload["mentioned_products"])
                chat.side_effect = grounded_reply
                response = answer_planning_question(question, self.context, "local", "model")
                chat.assert_called_once()
                self.assertNotEqual(response, SUPPORTED_QUESTION_FALLBACK)
                for name in route_planning_question(question, self.context)["context"]["mentioned_products"]:
                    self.assertIn(name, response)
                    self.assertIn(f"{self.inventory.loc[name, 'order_up_to_target']:,.0f}", response)

    def test_general_comparison_aliases_typos_and_ambiguity(self):
        for question in ["Compare yogurt and Sour Cream", "Compare Greek Yogrt and Sour Cream"]:
            route = route_planning_question(question, self.context)
            self.assertEqual(route["category"], "planning_comparison")
            self.assertIn("Greek Yogurt", route["reply"])
            self.assertIn("Sour Cream", route["reply"])
        with patch("ai_assistant.request_ollama_chat") as chat:
            response = answer_planning_question("Compare milk and yogurt", self.context, "local", "model")
            self.assertIn("Which product", response)
            chat.assert_not_called()

    def test_fallback_offline_returns_actual_states(self):
        from urllib.error import URLError
        with patch("ai_assistant.request_ollama_chat", side_effect=URLError("offline")):
            response = answer_planning_question("Explain the planning states of Sour Cream and Whole Milk.", self.context, "local", "model")
        self.assertIn("explanation service is unavailable", response)
        self.assertIn("Sour Cream", response)
        self.assertIn("Whole Milk", response)
        self.assertIn("Run down 712 units", response)
        self.assertIn("1,087", response)

    def test_missing_supplier_and_holiday_information_are_explicit(self):
        for question, expected in [
            ("What supplier should I use for Greek Yogurt?", "does not contain supplier or vendor information"),
            ("How much does Whole Milk cost?", "does not contain cost, pricing, or profit information"),
            ("Will Whole Milk demand increase next Christmas?", "cannot support a claim"),
        ]:
            with patch("ai_assistant.request_ollama_chat") as chat:
                response = answer_planning_question(question, self.context, "local", "model")
                self.assertIn(expected, response)
                self.assertNotEqual(response, SUPPORTED_QUESTION_FALLBACK)
                chat.assert_not_called()

    def test_requested_summary_remains_deterministic(self):
        response = self.answer("Give me a summary of the inventory situation.")
        self.assertIn("4 products", response)
        self.assertIn("Sour Cream: Run down 712 units", response)

    def test_fallback_prompt_has_grounding_rules(self):
        import json
        from ai_assistant import request_ollama_chat
        payload = route_planning_question("Explain the planning states of Sour Cream and Whole Milk.", self.context)["context"]
        with patch("ai_assistant.request.urlopen") as connection:
            connection.return_value.__enter__.return_value.read.return_value = b'{"message":{"content":"Grounded explanation"}}'
            request_ollama_chat("http://localhost:11434", "model", payload, [])
            request_body = json.loads(connection.call_args.args[0].data)
        self.assertEqual(request_body["options"], {"temperature": 0, "num_ctx": 8192})
        self.assertEqual(sum(message["role"] == "system" for message in request_body["messages"]), 1)
        instructions = request_body["messages"][0]["content"]
        self.assertIn("Current app context:", instructions)
        self.assertIn("factual_brief", instructions)
        for rule in ["only source of factual", "Never invent a numeric value", "Never create a new inventory target",
                     "Clearly distinguish facts from interpretation", "current analysis does not contain",
                     "Do not refuse merely because wording", "priority_order", "SAME excess"]:
            self.assertIn(rule, instructions)

    def test_assessment_and_metric_intent_precedence(self):
        cases = [
            ("Which product has the highest inventory?", "ranking"),
            ("Which product has the most current inventory?", "ranking"),
            ("Which product has the most on-hand stock?", "ranking"),
            ("Which product needs the most attention?", "prioritization"),
            ("What should I focus on first?", "prioritization"),
            ("Looking at the results overall, what are the biggest inventory problems?", "summary"),
            ("Looking at the current results overall, what are the biggest inventory problems you see and what would you focus on first?", "summary"),
            ("Assess the main inventory concerns and forecast accuracy overall", "summary"),
            ("What are the most important inventory issues?", "summary"),
            ("Give me a quick inventory summary.", "summary"),
            ("Looking at the results overall, what stands out?", "summary"),
            ("Compare Whole Milk and Greek Yogurt.", "planning_comparison"),
        ]
        for question, category in cases:
            with self.subTest(question=question):
                self.assertEqual(route_planning_question(question, self.context)["category"], category)
                answer = self.answer(question)
                self.assertNotEqual(answer, SUPPORTED_QUESTION_FALLBACK)
                if category == "ranking":
                    self.assertTrue(answer.startswith("Sour Cream — highest current inventory: 1,800"))
                if category == "summary":
                    self.assertTrue(answer.startswith("Greek Yogurt is the first planning concern"))
                    self.assertNotIn("highest current inventory", answer)
                    self.assertIn("Sour Cream: Run down 712 units", answer)

    def test_assessment_uses_all_products_and_existing_quality_labels(self):
        row = deepcopy(self.context["products_by_name"]["Greek Yogurt"])
        row.update(product="New Product", priority_score=3, pre_arrival_shortage=9999,
                   forecast_quality="Low confidence", product_mape=24.5)
        self.context["products_by_name"]["New Product"] = row
        answer = self.answer("What are the biggest inventory problems overall?")
        self.assertTrue(answer.startswith("New Product is the first planning concern"))
        self.assertIn("Pre-arrival service gap: 9,999 units", answer)
        self.assertIn("holdout MAPE 24.50%", answer)
        self.assertIn("Across 5 products", answer)
        self.assertNotRegex(answer.lower(), r"spoilage|supplier|popularity|shipment")

    def test_assessment_reports_missing_inputs_without_inventing_priority(self):
        for row in self.context["products_by_name"].values():
            row.update(priority_score=None, pre_arrival_shortage=None, recommended_action=None,
                       recommended_quantity_adjustment=None, planning_status="unavailable")
        answer = self.answer("Give me a quick inventory summary.")
        self.assertIn("4 products lack valid recommendations", answer)
        self.assertIn("Calculated priorities are unavailable", answer)
        self.assertNotIn("first planning concern", answer)

    def test_assessment_priority_ties_are_deterministic(self):
        for row in self.context["products_by_name"].values():
            row.update(priority_score=3, pre_arrival_shortage=50, recommended_action="Increase", recommended_quantity_adjustment=10)
        answer = self.answer("What are the biggest inventory problems?")
        self.assertTrue(answer.startswith("Greek Yogurt is the first planning concern"))
        self.assertIn("Focus first on Greek Yogurt, then Skim Milk, then Sour Cream", answer)


if __name__ == "__main__":
    unittest.main()
