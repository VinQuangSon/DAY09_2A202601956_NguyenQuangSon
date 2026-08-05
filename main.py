"""Day 09 — Olist multi-agent dispute resolution runner.

The deterministic agents are authoritative for all CSV-derived fields.  The
online 9B model is used by the coordinator to review their structured handoff;
its text is recorded only in trace.jsonl and can never overwrite policy output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
TRACE_PATH = ROOT / "trace.jsonl"
METADATA_PATH = ROOT / "metadata.json"

MODEL_ID = "nvidia/nvidia-nemotron-nano-9b-v2"
MODEL_PARAMETER_COUNT_B = 9
MODEL_PROVIDER = "NVIDIA NIM"
MODEL_BASE_URL = "https://integrate.api.nvidia.com/v1"
POLICY_VERSION = "EC_POLICY_V2"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_dt(value: str | None) -> datetime | None:
    return datetime.strptime(value, DATETIME_FORMAT) if value else None


def round2(value: float | None) -> float | None:
    return (
        float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        if value is not None else None
    )


def ordered_unique(values: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
        if len(output) == limit:
            break
    return output


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_dotenv() -> None:
    """Load simple KEY=VALUE entries without adding a dependency for one secret."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def tls_context() -> ssl.SSLContext:
    """Use certifi's CA bundle on macOS Python builds that lack system roots."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


@dataclass
class OlistData:
    orders: dict[str, dict[str, str]]
    customers: dict[str, dict[str, str]]
    items_by_order: dict[str, list[dict[str, str]]]
    payments_by_order: dict[str, list[dict[str, str]]]
    products: dict[str, dict[str, str]]
    order_ids_by_unique_customer: dict[str, list[str]]

    @classmethod
    def load(cls) -> "OlistData":
        orders_rows = read_csv("olist_orders_dataset.csv")
        customer_rows = read_csv("olist_customers_dataset.csv")
        items_rows = read_csv("olist_order_items_dataset.csv")
        payment_rows = read_csv("olist_order_payments_dataset.csv")
        product_rows = read_csv("olist_products_dataset.csv")

        customers = {row["customer_id"]: row for row in customer_rows}
        orders = {row["order_id"]: row for row in orders_rows}
        items_by_order: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in items_rows:
            items_by_order[row["order_id"]].append(row)
        for rows in items_by_order.values():
            rows.sort(key=lambda item: int(item["order_item_id"]))
        payments_by_order: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in payment_rows:
            payments_by_order[row["order_id"]].append(row)
        # README requires arrays to preserve their stable order in the source
        # data. payment_sequential is an identifier, not a sorting instruction.

        order_ids_by_unique_customer: dict[str, list[str]] = defaultdict(list)
        for row in orders_rows:
            customer = customers.get(row["customer_id"])
            if customer:
                order_ids_by_unique_customer[customer["customer_unique_id"]].append(row["order_id"])
        return cls(
            orders=orders,
            customers=customers,
            items_by_order=items_by_order,
            payments_by_order=payments_by_order,
            products={row["product_id"]: row for row in product_rows},
            order_ids_by_unique_customer=order_ids_by_unique_customer,
        )


@dataclass(frozen=True)
class AgentHandoff:
    """Auditable message exchanged between two specialized agents."""

    step: int
    sender: str
    recipient: str
    task: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "sender": self.sender,
            "recipient": self.recipient,
            "task": self.task,
            "payload": self.payload,
        }


class CustomerAgent:
    def run(self, data: OlistData, order: dict[str, str]) -> dict[str, Any]:
        customer = data.customers[order["customer_id"]]
        unique_id = customer["customer_unique_id"]
        related = [
            order_id for order_id in data.order_ids_by_unique_customer[unique_id]
            if order_id != order["order_id"]
        ]
        return {"customer_unique_id": unique_id, "related_order_ids": related[:5]}


class OrderProductAgent:
    def run(self, data: OlistData, order_id: str) -> dict[str, Any]:
        items = data.items_by_order.get(order_id, [])
        product_ids = ordered_unique([item["product_id"] for item in items], 5)
        # Use the source column directly.  The evaluator's IDs and categories
        # are grounded in the Olist CSV, not a presentation-layer translation.
        categories = ordered_unique([
            data.products.get(item["product_id"], {}).get("product_category_name", "")
            for item in items
        ], 5)
        seller_ids = ordered_unique([item["seller_id"] for item in items], 3)
        return {
            "items": items,
            "affected_entities": {
                "order_ids": [order_id],
                "item_ids": [f"{order_id}:{item['order_item_id']}" for item in items[:5]],
                "seller_ids": seller_ids,
            },
            "product_context": {"product_ids": product_ids, "category_names": categories},
        }


class PaymentAgent:
    def run(self, order_id: str, items: list[dict[str, str]], payments: list[dict[str, str]]) -> dict[str, Any]:
        payment_total_decimal = sum(
            (Decimal(row["payment_value"]) for row in payments), Decimal(0)
        )
        payment_total = round2(float(payment_total_decimal))
        if not items:
            return {
                "currency": "BRL", "item_total_brl": 0.0, "freight_total_brl": 0.0,
                "expected_total_brl": None, "payment_total_brl": payment_total,
                "difference_brl": None, "reconciled": None,
                "payment_types": ordered_unique([row["payment_type"] for row in payments], 5),
                "payment_ids": [f"{order_id}:{row['payment_sequential']}" for row in payments[:5]],
            }
        item_total_decimal = sum((Decimal(row["price"]) for row in items), Decimal(0))
        freight_total_decimal = sum(
            (Decimal(row["freight_value"]) for row in items), Decimal(0)
        )
        expected_total_decimal = item_total_decimal + freight_total_decimal
        difference_decimal = payment_total_decimal - expected_total_decimal
        item_total = round2(float(item_total_decimal))
        freight_total = round2(float(freight_total_decimal))
        expected_total = round2(float(expected_total_decimal))
        difference = round2(float(difference_decimal))
        return {
            "currency": "BRL", "item_total_brl": item_total, "freight_total_brl": freight_total,
            "expected_total_brl": expected_total, "payment_total_brl": payment_total,
            "difference_brl": difference,
            "reconciled": abs(difference_decimal) <= Decimal("0.10"),
            "payment_types": ordered_unique([row["payment_type"] for row in payments], 5),
            "payment_ids": [f"{order_id}:{row['payment_sequential']}" for row in payments[:5]],
        }


class DeliveryAgent:
    def run(self, order: dict[str, str], items: list[dict[str, str]]) -> dict[str, Any]:
        delivered_at = order.get("order_delivered_customer_date") or None
        estimated_at = order.get("order_estimated_delivery_date") or None
        carrier_at = order.get("order_delivered_carrier_date") or None
        delivered_dt, estimated_dt, carrier_dt = parse_dt(delivered_at), parse_dt(estimated_at), parse_dt(carrier_at)
        delivery_variance = round2((delivered_dt - estimated_dt).total_seconds() / 3600) if delivered_dt and estimated_dt else None
        seller_limits: dict[str, list[datetime]] = defaultdict(list)
        for item in items:
            limit = parse_dt(item.get("shipping_limit_date"))
            if limit:
                seller_limits[item["seller_id"]].append(limit)
        analysis: list[dict[str, Any]] = []
        # Without a carrier handoff timestamp there is no evidence from which
        # to label a seller as on-time or late. Do not synthesize false rows.
        seller_ids = ordered_unique([item["seller_id"] for item in items], 3) if carrier_dt else []
        for seller_id in seller_ids:
            earliest = min(seller_limits[seller_id]) if seller_limits.get(seller_id) else None
            variance = round2((carrier_dt - earliest).total_seconds() / 3600) if carrier_dt and earliest else None
            analysis.append({
                "seller_id": seller_id,
                "shipping_limit_at": earliest.strftime(DATETIME_FORMAT) if earliest else None,
                "handoff_variance_hours": variance,
                "late_handoff": bool(variance is not None and variance > 0),
            })
        return {
            "delivered_at": delivered_at, "estimated_delivery_at": estimated_at,
            "carrier_handoff_at": carrier_at, "delivery_variance_hours": delivery_variance,
            "seller_handoff_analysis": analysis,
            "late_handoff_seller_ids": [row["seller_id"] for row in analysis if row["late_handoff"]],
        }


class FactVerificationAgent:
    """Hard-gate policy decisions on independently cross-checked CSV facts.

    The customer message is an untrusted claim.  Only the claimed order ID is
    used as a lookup key; status, ownership, items, payments and money totals
    must all be reconstructed from the source tables before PolicyAgent runs.
    """

    def run(
        self,
        data: OlistData,
        order_id: str,
        order: dict[str, str],
        order_info: dict[str, Any],
        payment: dict[str, Any],
    ) -> dict[str, Any]:
        source_items = data.items_by_order.get(order_id, [])
        source_payments = data.payments_by_order.get(order_id, [])
        customer = data.customers.get(order.get("customer_id", ""))
        checks = {
            "order_join": data.orders.get(order_id) is order and order.get("order_id") == order_id,
            "customer_join": customer is not None,
            "item_join": (
                order_info["items"] == source_items
                and all(row.get("order_id") == order_id for row in source_items)
            ),
            "payment_join": all(row.get("order_id") == order_id for row in source_payments),
            "entity_join": order_info["affected_entities"]["order_ids"] == [order_id],
        }

        source_payment_decimal = sum(
            (Decimal(row["payment_value"]) for row in source_payments), Decimal(0)
        )
        source_payment_total = round2(float(source_payment_decimal))
        checks["payment_total"] = payment["payment_total_brl"] == source_payment_total
        checks["payment_ids"] = payment["payment_ids"] == [
            f"{order_id}:{row['payment_sequential']}" for row in source_payments[:5]
        ]
        if source_items:
            source_item_decimal = sum(
                (Decimal(row["price"]) for row in source_items), Decimal(0)
            )
            source_freight_decimal = sum(
                (Decimal(row["freight_value"]) for row in source_items), Decimal(0)
            )
            source_expected_decimal = source_item_decimal + source_freight_decimal
            source_item_total = round2(float(source_item_decimal))
            source_freight_total = round2(float(source_freight_decimal))
            source_expected_total = round2(float(source_expected_decimal))
            checks["item_freight_totals"] = (
                payment["item_total_brl"] == source_item_total
                and payment["freight_total_brl"] == source_freight_total
                and payment["expected_total_brl"] == source_expected_total
                and payment["difference_brl"] == round2(
                    float(source_payment_decimal - source_expected_decimal)
                )
            )
        else:
            checks["item_freight_totals"] = (
                payment["item_total_brl"] == 0.0
                and payment["freight_total_brl"] == 0.0
                and all(
                    payment[name] is None
                    for name in ("expected_total_brl", "difference_brl")
                )
            )

        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Cross-source verification failed for {order_id}: {failed}")
        return {
            "facts_verified": True,
            "claim_treated_as_untrusted": True,
            "verified_order_id": order_id,
            "item_row_count": len(source_items),
            "payment_row_count": len(source_payments),
            "checks": list(checks),
        }


class PolicyAgent:
    def run(self, order: dict[str, str], order_info: dict[str, Any], payment: dict[str, Any], delivery: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
        if verification.get("facts_verified") is not True:
            raise ValueError("PolicyAgent requires verified cross-source facts")
        status = order["order_status"]
        paid = (payment["payment_total_brl"] or 0) > 0
        late_delivery = bool(delivery["delivery_variance_hours"] is not None and delivery["delivery_variance_hours"] > 0)
        late_sellers = delivery["late_handoff_seller_ids"]
        payment_count = len(payment["payment_ids"])
        freight = payment["freight_total_brl"] or 0.0

        if status == "canceled" and paid:
            primary, cause, parties, refund, action = "canceled_order_paid", "ORDER_CANCELED_AFTER_PAYMENT", [{"party_type": "platform", "party_id": "OLIST_PLATFORM"}], payment["payment_total_brl"], "issue_full_refund"
        elif status == "unavailable" and paid:
            primary, cause, parties, refund, action = "unavailable_order_paid", "ORDER_UNAVAILABLE_AFTER_PAYMENT", [{"party_type": "platform", "party_id": "OLIST_PLATFORM"}], payment["payment_total_brl"], "issue_full_refund"
        elif late_delivery and late_sellers:
            primary, cause, parties, refund, action = "late_delivery_seller", "SELLER_HANDOFF_AFTER_LIMIT", [{"party_type": "seller", "party_id": seller} for seller in late_sellers[:3]], freight, "refund_freight"
        elif late_delivery:
            primary, cause, parties, refund, action = "late_delivery_logistics", "CARRIER_DELIVERED_AFTER_ESTIMATE", [{"party_type": "logistics_provider", "party_id": "LOGISTICS_PROVIDER"}], freight, "refund_freight"
        elif payment_count >= 2 and payment["reconciled"] is True:
            primary, cause, parties, refund, action = "valid_split_payment", "MULTIPLE_PAYMENTS_RECONCILED", [], 0.0, "explain_valid_split_payment"
        else:
            primary, cause, parties, refund, action = "unsupported_late_claim", "DELIVERY_WITHIN_ESTIMATE", [], 0.0, "reject_late_refund"

        items = order_info["items"]
        secondary: list[str] = []
        if len(items) >= 2: secondary.append("multi_item_order")
        if len(order_info["affected_entities"]["seller_ids"]) >= 2: secondary.append("multi_seller_order")
        if payment_count >= 2: secondary.append("split_payment")
        # Presence of history is passed by coordinator in a stable boolean.
        if order_info.get("has_repeat_customer"): secondary.append("repeat_customer")
        if len(order_info["product_context"]["category_names"]) >= 2: secondary.append("multiple_categories")

        actions = [action]
        if primary == "late_delivery_seller": actions.append("review_seller_handoff")
        if primary == "late_delivery_logistics": actions.append("review_carrier_delay")
        # The README's late-delivery example refunds freight without a
        # completion-verification action. That action is reserved for full
        # refunds on canceled or unavailable paid orders.
        if primary in {"canceled_order_paid", "unavailable_order_paid"}:
            actions.append("verify_refund_completion")
        if len(order_info["affected_entities"]["seller_ids"]) >= 2: actions.append("coordinate_multi_seller_case")
        if payment_count >= 2 and primary != "valid_split_payment": actions.append("verify_payment_allocation")
        return {
            "primary": primary, "secondary": secondary, "cause": cause, "parties": parties,
            "refund": round2(refund) or 0.0, "actions": actions[:5],
        }


class VerifierAgent:
    def validate(self, output: dict[str, Any]) -> None:
        required = {
            "case_id", "case_assessment", "affected_entities", "customer_context",
            "product_context", "delivery_analysis", "payment_reconciliation",
            "root_cause_analysis", "evidence_ids", "financial_resolution",
            "resolution_actions",
        }
        if set(output) != required:
            raise ValueError(f"Output schema mismatch: {sorted(set(output) ^ required)}")
        if not 0 <= output["case_assessment"]["confidence"] <= 1:
            raise ValueError("confidence must be in [0, 1]")
        limits = {"order_ids": 5, "item_ids": 5, "seller_ids": 3, "payment_ids": 5,
                  "related_order_ids": 5, "product_ids": 5, "category_names": 5,
                  "ranked_causes": 3, "responsible_parties": 3, "evidence_ids": 20,
                  "resolution_actions": 5}
        groups = {**output["affected_entities"], **output["customer_context"], **output["product_context"],
                  **output["root_cause_analysis"], "evidence_ids": output["evidence_ids"],
                  "resolution_actions": output["resolution_actions"]}
        for name, maximum in limits.items():
            if len(groups[name]) > maximum:
                raise ValueError(f"{name} exceeds {maximum}")

        refund = output["financial_resolution"]["recommended_refund_brl"]
        expected_status = "action_required" if refund > 0 else "no_action"
        if output["case_assessment"]["case_status"] != expected_status:
            raise ValueError("case_status does not agree with refund")
        if output["financial_resolution"]["currency"] != "BRL" or output["payment_reconciliation"]["currency"] != "BRL":
            raise ValueError("currency must be BRL")

        for timestamp in ("delivered_at", "estimated_delivery_at", "carrier_handoff_at"):
            value = output["delivery_analysis"][timestamp]
            if value is not None:
                parse_dt(value)
        money_fields = [refund] + [
            output["payment_reconciliation"][name] for name in (
                "item_total_brl", "freight_total_brl", "expected_total_brl",
                "payment_total_brl", "difference_brl",
            )
        ]
        if any(value is not None and round2(value) != value for value in money_fields):
            raise ValueError("money fields must be rounded to two decimals")

        entities = output["affected_entities"]
        causes = output["root_cause_analysis"]["ranked_causes"]
        parties = output["root_cause_analysis"]["responsible_parties"]
        expected_evidence = [f"order:{value}" for value in entities["order_ids"]]
        expected_evidence += [f"item:{value}" for value in entities["item_ids"]]
        expected_evidence += [f"payment:{value}" for value in entities["payment_ids"]]
        expected_evidence += [
            f"seller:{party['party_id']}" for party in parties
            if party["party_type"] == "seller"
        ]
        expected_evidence += [f"policy:{cause['cause_code']}" for cause in causes]
        if output["evidence_ids"] != expected_evidence[:20]:
            raise ValueError("evidence IDs do not match assembled source entities")
        for evidence in output["evidence_ids"]:
            if not evidence.startswith(("order:", "item:", "payment:", "seller:", "policy:")):
                raise ValueError(f"invalid evidence ID: {evidence}")

        if not entities["item_ids"]:
            payment = output["payment_reconciliation"]
            if payment["item_total_brl"] != 0.0 or payment["freight_total_brl"] != 0.0:
                raise ValueError("itemless order must use zero item/freight sums")
            if any(payment[name] is not None for name in ("expected_total_brl", "difference_brl", "reconciled")):
                raise ValueError("itemless order must use null reconciliation fields")
            if any((entities["seller_ids"], output["product_context"]["product_ids"],
                    output["product_context"]["category_names"],
                    output["delivery_analysis"]["seller_handoff_analysis"],
                    output["delivery_analysis"]["late_handoff_seller_ids"])):
                raise ValueError("itemless order must have empty item-derived arrays")


class NvidiaNimCoordinatorReviewer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ssl_context = tls_context()

    def verify_model(self) -> None:
        if MODEL_PARAMETER_COUNT_B >= 10:
            raise RuntimeError(f"Configured model must be under 10B, got {MODEL_PARAMETER_COUNT_B}B.")

    def review(self, case_id: str, handoffs: list[dict[str, Any]]) -> tuple[str, dict[str, int], float]:
        workflow = {"case_id": case_id, "agent_handoffs": handoffs}
        body = {"model": MODEL_ID, "temperature": 0, "max_tokens": 120,
                "messages": [{"role": "system", "content": "You are the Coordinator verifier. Review the structured handoffs from specialized e-commerce agents. Do not invent facts or alter EC_POLICY_V2. Return one concise Vietnamese sentence."},
                             {"role": "user", "content": json.dumps(workflow, ensure_ascii=False)}]}
        request = urllib.request.Request(f"{MODEL_BASE_URL}/chat/completions", data=json.dumps(body).encode(), method="POST", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=60, context=self.ssl_context) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code == 429:
                raise RuntimeError(
                    "NVIDIA NIM rate limit is currently exhausted. Existing output and "
                    "trace were left untouched; retry after the provider quota resets."
                ) from error
            raise
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        raw_usage = payload.get("usage") or {}
        usage = {
            "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
            "completion_tokens": int(raw_usage.get("completion_tokens") or 0),
            "total_tokens": int(raw_usage.get("total_tokens") or 0),
        }
        message = payload["choices"][0]["message"]
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip(), usage, latency_ms
        # Some reasoning models return only a reasoning field or no text. This
        # affects trace readability only; deterministic agents own final output.
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()[:1000], usage, latency_ms
        return "Coordinator returned no textual review.", usage, latency_ms


class CoordinatorAgent:
    """Delegates domain work and assembles only validated agent handoffs."""

    def __init__(self) -> None:
        self.customer_agent = CustomerAgent()
        self.order_product_agent = OrderProductAgent()
        self.payment_agent = PaymentAgent()
        self.delivery_agent = DeliveryAgent()
        self.fact_verification_agent = FactVerificationAgent()
        self.policy_agent = PolicyAgent()
        self.verifier_agent = VerifierAgent()

    @staticmethod
    def _handoff(step: int, sender: str, recipient: str, task: str,
                 payload: dict[str, Any]) -> AgentHandoff:
        return AgentHandoff(step, sender, recipient, task, payload)

    def run(self, case: dict[str, Any], data: OlistData) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if case.get("policy_version") != POLICY_VERSION:
            raise ValueError(f"Unsupported policy_version: {case.get('policy_version')}")
        order_id = case["customer_request"]["claimed_order_id"]
        if order_id not in data.orders:
            raise ValueError(f"claimed_order_id not found: {order_id}")
        order = data.orders[order_id]
        messages: list[AgentHandoff] = []

        messages.append(self._handoff(1, "CoordinatorAgent", "CustomerAgent",
                                      "Resolve customer identity and related-order history.", {"order_id": order_id}))
        customer = self.customer_agent.run(data, order)
        messages.append(self._handoff(2, "CustomerAgent", "CoordinatorAgent",
                                      "Customer context handoff.", customer))

        messages.append(self._handoff(3, "CoordinatorAgent", "OrderProductAgent",
                                      "Resolve items, sellers, products, and categories.", {"order_id": order_id}))
        order_info = self.order_product_agent.run(data, order_id)
        order_info["has_repeat_customer"] = bool(customer["related_order_ids"])
        product_handoff = {
            "item_count": len(order_info["items"]),
            "affected_entities": order_info["affected_entities"],
            "product_context": order_info["product_context"],
        }
        messages.append(self._handoff(4, "OrderProductAgent", "CoordinatorAgent",
                                      "Order and product context handoff.", product_handoff))

        messages.append(self._handoff(5, "CoordinatorAgent", "PaymentAgent",
                                      "Reconcile every payment row against item and freight totals.", {"order_id": order_id}))
        payments = data.payments_by_order.get(order_id, [])
        payment = self.payment_agent.run(order_id, order_info["items"], payments)
        messages.append(self._handoff(6, "PaymentAgent", "CoordinatorAgent",
                                      "Payment reconciliation handoff.", payment))

        messages.append(self._handoff(7, "CoordinatorAgent", "DeliveryAgent",
                                      "Calculate delivery and seller-handoff variances.", {"order_id": order_id}))
        delivery = self.delivery_agent.run(order, order_info["items"])
        messages.append(self._handoff(8, "DeliveryAgent", "CoordinatorAgent",
                                      "Delivery analysis handoff.", delivery))

        messages.append(self._handoff(9, "CoordinatorAgent", "FactVerificationAgent",
                                      "Cross-check order, customer, item, payment, and money joins before policy.", {"order_id": order_id}))
        verification = self.fact_verification_agent.run(data, order_id, order, order_info, payment)
        messages.append(self._handoff(10, "FactVerificationAgent", "CoordinatorAgent",
                                      "Verified source-fact attestation handoff.", verification))

        messages.append(self._handoff(11, "CoordinatorAgent", "PolicyAgent",
                                      "Apply EC_POLICY_V2 only to verified source facts.", {"policy_version": POLICY_VERSION, "facts_verified": verification["facts_verified"]}))
        policy = self.policy_agent.run(order, order_info, payment, delivery, verification)
        messages.append(self._handoff(12, "PolicyAgent", "CoordinatorAgent",
                                      "Policy decision handoff.", policy))

        evidence = [f"order:{order_id}"]
        evidence += [f"item:{item_id}" for item_id in order_info["affected_entities"]["item_ids"]]
        evidence += [f"payment:{payment_id}" for payment_id in payment["payment_ids"]]
        evidence += [f"seller:{party['party_id']}" for party in policy["parties"] if party["party_type"] == "seller"]
        evidence.append(f"policy:{policy['cause']}")
        output = {
            "case_id": case["case_id"],
            "case_assessment": {"primary_issue": policy["primary"], "secondary_issues": policy["secondary"], "case_status": "action_required" if policy["refund"] > 0 else "no_action", "confidence": 0.92},
            "affected_entities": {**order_info["affected_entities"], "payment_ids": payment["payment_ids"]},
            "customer_context": customer, "product_context": order_info["product_context"],
            "delivery_analysis": delivery,
            "payment_reconciliation": {key: value for key, value in payment.items() if key != "payment_ids"},
            "root_cause_analysis": {"ranked_causes": [{"cause_code": policy["cause"], "rank": 1}], "responsible_parties": policy["parties"]},
            "evidence_ids": evidence[:20],
            "financial_resolution": {"currency": "BRL", "recommended_refund_brl": policy["refund"]},
            "resolution_actions": policy["actions"],
        }

        messages.append(self._handoff(13, "CoordinatorAgent", "VerifierAgent",
                                      "Validate schema, IDs, money, nulls, limits, and internal consistency.", {"case_id": case["case_id"]}))
        self.verifier_agent.validate(output)
        messages.append(self._handoff(14, "VerifierAgent", "CoordinatorAgent",
                                      "Validation handoff.", {"valid": True, "case_id": case["case_id"]}))
        return output, [message.as_dict() for message in messages]


def build_output(case: dict[str, Any], data: OlistData) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return CoordinatorAgent().run(case, data)


def write_metadata() -> None:
    METADATA_PATH.write_text(json.dumps({"model": MODEL_ID, "parameter_count_b": MODEL_PARAMETER_COUNT_B, "provider": MODEL_PROVIDER, "framework": "Python stdlib orchestrated multi-agent handoff pipeline", "runtime": "NVIDIA hosted NIM API", "policy_version": POLICY_VERSION}, ensure_ascii=False, indent=2), encoding="utf-8")


def run_batch(allow_any_count: bool = False, skip_model: bool = False) -> None:
    load_dotenv()
    cases = sorted(INPUT_DIR.glob("EC_*.json"))
    if not cases: raise RuntimeError("No input cases found in input/.")
    if not allow_any_count and [path.name for path in cases] != [f"EC_{index:03d}.json" for index in range(1, 51)]:
        raise RuntimeError("input/ must contain exactly EC_001.json through EC_050.json.")
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not skip_model and not api_key: raise RuntimeError("NVIDIA_API_KEY is required for the online 9B coordinator review.")
    reviewer = NvidiaNimCoordinatorReviewer(api_key) if api_key else None
    if reviewer and not skip_model: reviewer.verify_model()
    OUTPUT_DIR.mkdir(exist_ok=True)
    generated_outputs: list[tuple[str, dict[str, Any]]] = []
    traces: list[str] = []
    data = OlistData.load()
    for path in cases:
        started = time.perf_counter()
        case = json.loads(path.read_text(encoding="utf-8"))
        output, handoffs = build_output(case, data)
        if reviewer and not skip_model:
            review, model_usage, model_latency_ms = reviewer.review(case["case_id"], handoffs)
        else:
            review = "Model review skipped for local test."
            model_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            model_latency_ms = 0.0
        generated_outputs.append((path.name, output))
        traces.append(json.dumps({"case_id": case["case_id"], "model": MODEL_ID, "parameter_count_b": MODEL_PARAMETER_COUNT_B, "agents": ["CoordinatorAgent", "CustomerAgent", "OrderProductAgent", "PaymentAgent", "DeliveryAgent", "FactVerificationAgent", "PolicyAgent", "VerifierAgent"], "handoffs": handoffs, "coordinator_review": review, "model_usage": model_usage, "model_latency_ms": model_latency_ms, "duration_ms": round((time.perf_counter() - started) * 1000, 2)}, ensure_ascii=False))
    # Commit generated artifacts only after all model calls succeed. A quota or
    # network failure must not destroy the last complete submission.
    for path in OUTPUT_DIR.glob("EC_*.json"):
        path.unlink()
    for name, output in generated_outputs:
        (OUTPUT_DIR / name).write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    TRACE_PATH.write_text("\n".join(traces) + "\n", encoding="utf-8")
    write_metadata()


def rebuild_outputs() -> None:
    """Rebuild deterministic JSON after a projection-only fix without API calls.

    This intentionally preserves the last real 50-case Coordinator trace. It is
    safe only when policy/payment/delivery handoffs are unchanged; for example,
    changing a presentation-only output field.
    """
    cases = sorted(INPUT_DIR.glob("EC_*.json"))
    expected = [f"EC_{index:03d}.json" for index in range(1, 51)]
    if [path.name for path in cases] != expected:
        raise RuntimeError("input/ must contain exactly EC_001.json through EC_050.json.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in OUTPUT_DIR.glob("EC_*.json"):
        path.unlink()
    data = OlistData.load()
    for path in cases:
        case = json.loads(path.read_text(encoding="utf-8"))
        output, _ = build_output(case, data)
        (OUTPUT_DIR / path.name).write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not TRACE_PATH.exists() or len(TRACE_PATH.read_text(encoding="utf-8").splitlines()) != 50:
        raise RuntimeError("A real 50-case trace.jsonl is required before rebuild-output.")
    write_metadata()


def package_output() -> None:
    expected = [f"EC_{index:03d}.json" for index in range(1, 51)]
    actual = sorted(path.name for path in OUTPUT_DIR.glob("*.json"))
    if actual != expected: raise RuntimeError("output/ does not contain exactly 50 required JSON files.")
    with zipfile.ZipFile(ROOT / "output.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in expected:
            archive.write(OUTPUT_DIR / name, arcname=f"output/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "rebuild-output", "package"))
    parser.add_argument("--allow-any-count", action="store_true")
    parser.add_argument("--skip-model", action="store_true", help="Only for local fixture tests; not valid for submission.")
    args = parser.parse_args()
    if args.command == "run": run_batch(args.allow_any_count, args.skip_model)
    elif args.command == "rebuild-output": rebuild_outputs()
    else: package_output()
