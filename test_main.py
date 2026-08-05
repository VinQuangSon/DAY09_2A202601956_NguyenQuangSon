import unittest
import json

from main import FactVerificationAgent, INPUT_DIR, OlistData, PaymentAgent, PolicyAgent, build_output


class PolicyPipelineTests(unittest.TestCase):
    def test_coordinator_records_real_agent_handoffs(self):
        case = json.loads((INPUT_DIR / "EC_001.json").read_text(encoding="utf-8"))
        output, handoffs = build_output(case, OlistData.load())
        self.assertEqual(output["case_id"], "EC_001")
        self.assertEqual(output["case_assessment"]["confidence"], 0.92)
        self.assertEqual(len(handoffs), 14)
        self.assertEqual(handoffs[0]["sender"], "CoordinatorAgent")
        self.assertEqual(handoffs[0]["recipient"], "CustomerAgent")
        self.assertEqual(handoffs[-1]["sender"], "VerifierAgent")
        self.assertTrue(handoffs[-1]["payload"]["valid"])
        returned_agents = {
            row["sender"] for row in handoffs if row["recipient"] == "CoordinatorAgent"
        }
        self.assertEqual(returned_agents, {
            "CustomerAgent", "OrderProductAgent", "PaymentAgent",
            "DeliveryAgent", "FactVerificationAgent", "PolicyAgent", "VerifierAgent",
        })
        fact_handoff = next(row for row in handoffs if row["sender"] == "FactVerificationAgent")
        self.assertTrue(fact_handoff["payload"]["facts_verified"])
        self.assertTrue(fact_handoff["payload"]["claim_treated_as_untrusted"])

    def test_payment_rows_preserve_csv_source_order(self):
        data = OlistData.load()
        order_id = "23c312ca9f0242a48a95e5643bee2645"
        self.assertEqual(
            [row["payment_sequential"] for row in data.payments_by_order[order_id]],
            ["2", "1"],
        )

    def test_reconciled_split_payment(self):
        payment = PaymentAgent().run("o1", [{"price": "90", "freight_value": "10"}], [
            {"payment_value": "50", "payment_sequential": "1", "payment_type": "voucher"},
            {"payment_value": "50", "payment_sequential": "2", "payment_type": "credit_card"},
        ])
        self.assertTrue(payment["reconciled"])
        policy = PolicyAgent().run({"order_status": "delivered"}, {
            "items": [{"seller_id": "s1"}], "affected_entities": {"seller_ids": ["s1"]},
            "product_context": {"category_names": ["books"]}, "has_repeat_customer": False,
        }, payment, {"delivery_variance_hours": -2.0, "late_handoff_seller_ids": []}, {"facts_verified": True})
        self.assertEqual(policy["primary"], "valid_split_payment")
        self.assertNotIn("verify_payment_allocation", policy["actions"])

    def test_seller_late_has_freight_refund(self):
        payment = {"payment_total_brl": 120.0, "freight_total_brl": 12.5, "reconciled": True, "payment_ids": ["o1:1"]}
        policy = PolicyAgent().run({"order_status": "delivered"}, {
            "items": [{"seller_id": "s1"}], "affected_entities": {"seller_ids": ["s1"]},
            "product_context": {"category_names": ["books"]}, "has_repeat_customer": False,
        }, payment, {"delivery_variance_hours": 1.0, "late_handoff_seller_ids": ["s1"]}, {"facts_verified": True})
        self.assertEqual(policy["primary"], "late_delivery_seller")
        self.assertEqual(policy["refund"], 12.5)
        self.assertEqual(policy["parties"][0]["party_id"], "s1")
        self.assertNotIn("verify_refund_completion", policy["actions"])

    def test_full_refund_requires_completion_verification(self):
        payment = {
            "payment_total_brl": 120.0,
            "freight_total_brl": None,
            "reconciled": None,
            "payment_ids": ["o1:1"],
        }
        policy = PolicyAgent().run({"order_status": "canceled"}, {
            "items": [], "affected_entities": {"seller_ids": []},
            "product_context": {"category_names": []}, "has_repeat_customer": False,
        }, payment, {"delivery_variance_hours": None, "late_handoff_seller_ids": []}, {"facts_verified": True})
        self.assertEqual(policy["primary"], "canceled_order_paid")
        self.assertIn("verify_refund_completion", policy["actions"])

    def test_itemless_order_uses_zero_sums_and_null_reconciliation(self):
        payment = PaymentAgent().run("o1", [], [{
            "payment_value": "42.50", "payment_sequential": "1",
            "payment_type": "credit_card", "order_id": "o1",
        }])
        self.assertEqual(payment["item_total_brl"], 0.0)
        self.assertEqual(payment["freight_total_brl"], 0.0)
        self.assertIsNone(payment["expected_total_brl"])
        self.assertIsNone(payment["difference_brl"])
        self.assertIsNone(payment["reconciled"])

    def test_missing_carrier_does_not_synthesize_seller_handoff(self):
        from main import DeliveryAgent
        delivery = DeliveryAgent().run(
            {
                "order_delivered_customer_date": "",
                "order_estimated_delivery_date": "2018-05-10 00:00:00",
                "order_delivered_carrier_date": "",
            },
            [{"seller_id": "s1", "shipping_limit_date": "2018-05-01 00:00:00"}],
        )
        self.assertEqual(delivery["seller_handoff_analysis"], [])
        self.assertEqual(delivery["late_handoff_seller_ids"], [])

    def test_policy_rejects_unverified_customer_claim(self):
        with self.assertRaisesRegex(ValueError, "verified cross-source facts"):
            PolicyAgent().run(
                {"order_status": "canceled"},
                {"items": [], "affected_entities": {"seller_ids": []},
                 "product_context": {"category_names": []}, "has_repeat_customer": False},
                {"payment_total_brl": 100.0, "freight_total_brl": None,
                 "reconciled": None, "payment_ids": ["o1:1"]},
                {"delivery_variance_hours": None, "late_handoff_seller_ids": []},
                {"facts_verified": False},
            )

    def test_fact_verifier_detects_tampered_payment_total(self):
        data = OlistData.load()
        case = json.loads((INPUT_DIR / "EC_001.json").read_text(encoding="utf-8"))
        order_id = case["customer_request"]["claimed_order_id"]
        order = data.orders[order_id]
        from main import OrderProductAgent
        order_info = OrderProductAgent().run(data, order_id)
        payment = PaymentAgent().run(order_id, order_info["items"], data.payments_by_order[order_id])
        payment["payment_total_brl"] += 1
        with self.assertRaisesRegex(ValueError, "payment_total"):
            FactVerificationAgent().run(data, order_id, order, order_info, payment)


if __name__ == "__main__":
    unittest.main()
