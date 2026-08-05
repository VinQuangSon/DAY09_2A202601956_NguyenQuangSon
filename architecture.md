# Multi-Agent Olist Dispute Resolution

```text
Input case
   │
   ├─ Customer Agent ────── customer identity + related orders
   ├─ Order/Product Agent ─ items + sellers + products + categories
   ├─ Payment Agent ─────── payment reconciliation
   ├─ Delivery Agent ────── delivery and seller-handoff variance
   │
   └─ Policy Agent ─────── EC_POLICY_V2 decision
                               │
                    Coordinator (NVIDIA NIM Nemotron Nano 9B)
                    review-only trace; cannot modify result
                               │
                    Verifier Agent ─ schema, IDs, limits, nulls
                               │
                         output/EC_XXX.json + trace.jsonl
```

## Agent contracts and access

| Agent | Reads | Handoff | Authority |
| --- | --- | --- | --- |
| Customer | orders, customers | `customer_context` | Customer identity/history only |
| Order/Product | orders, items, products | entities, products, categories | CSV facts only |
| Payment | items, payments | reconciliation totals | CSV arithmetic only |
| Delivery | orders, items | timestamp variance, late seller IDs | CSV timestamps only |
| Policy | all previous handoffs | issue, cause, parties, refund, actions | `EC_POLICY_V2` only |
| Coordinator | compact handoffs | one Vietnamese review sentence in trace | Review only; no output mutation |
| Verifier | assembled output | accept/reject | schema, evidence IDs, limits, null rules |

The coordinator uses `nvidia/nvidia-nemotron-nano-9b-v2` through NVIDIA's hosted
NIM API. It is declared as 9B parameters in source and metadata. The runner rejects
any configured model size at or above 10B and never routes to a larger fallback.

## Runbook

```bash
cp .env.example .env
# Put NVIDIA_API_KEY in .env.
python3 main.py run
python3 main.py package
```

`run` requires exactly `input/EC_001.json` through `input/EC_050.json` and
replaces prior generated output and trace. `package` creates `output.zip` only
when `output/` contains exactly those 50 JSON files.

`rebuild-output` runs the deterministic rule-based agents without consuming API
quota and preserves the existing real 50-case trace. It is intended for an
output projection repair that does not change policy/payment/delivery handoffs.
