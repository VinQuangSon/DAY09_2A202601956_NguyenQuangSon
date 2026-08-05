# Multi-Agent Olist Dispute Resolution

```text
Input case
   │
Coordinator Agent
   ├─ task → Customer Agent ─────── identity + related orders ── handoff ─┐
   ├─ task → Order/Product Agent ── items/sellers/products ───── handoff ─┤
   ├─ task → Payment Agent ──────── reconciliation ───────────── handoff ─┤
   ├─ task → Delivery Agent ─────── variances/late sellers ───── handoff ─┤
   │                                                                    │
   ├─ task → Fact Verification Agent ← order/item/payment joins ─────────┘
   │              └─ hard gate: reject mismatched or ungrounded facts
   │
   ├─ task → Policy Agent ←──────── verified fact attestation only
   │              └─ taxonomy + responsibility + refund + actions
   │
   ├─ assemble candidate output
   ├─ task → Verifier Agent ─────── schema/IDs/money/nulls/limits
   │              └─ valid handoff or reject
   │
   └─ NVIDIA NIM Nemotron Nano 9B reviews the full handoff chain
                   └─ output/EC_XXX.json + one trace.jsonl record
```

## Agent contracts and access

| Agent | Reads | Handoff | Authority |
| --- | --- | --- | --- |
| Customer | orders, customers | `customer_context` | Customer identity/history only |
| Order/Product | orders, items, products | entities, products, categories | CSV facts only |
| Payment | items, payments | reconciliation totals | CSV arithmetic only |
| Delivery | orders, items | timestamp variance, late seller IDs | CSV timestamps only |
| Fact Verification | orders, customers, items, payments plus domain handoffs | cross-source attestation | May stop the case; cannot decide policy |
| Policy | verified handoffs + attestation | issue, cause, parties, refund, actions | `EC_POLICY_V2` only; refuses unverified input |
| Coordinator | input plus all returned handoffs | assignments, assembled candidate, one NIM review | Orchestration and assembly; no CSV arithmetic |
| Verifier | assembled output | accept/reject | schema, evidence IDs, limits, null rules |

The customer complaint is treated as untrusted input. Its `claimed_order_id` is
only a lookup key. Before Policy runs, Fact Verification independently joins the
order to its customer, every item row and every payment row, then recomputes the
payment and item/freight totals. A failed check raises an error before any refund
or resolution action can be produced.

Missing data is not converted into a negative fact. In particular, when
`order_delivered_carrier_date` is absent, Delivery returns empty seller-handoff
arrays instead of inventing an on-time handoff. Itemless orders use `0.0` for
the observable item/freight sums and `null` for expected total, difference and
reconciliation status. Money uses decimal `ROUND_HALF_UP` semantics.

Each case records 14 structured messages: seven Coordinator assignments and seven
agent returns. The trace therefore proves which agent performed each task and
what payload was handed back; the implementation does not place the whole case
into one monolithic prompt. The deterministic domain agents remain authoritative
for CSV facts, while NVIDIA NIM reviews the completed handoff chain without
overwriting it.

The coordinator uses `nvidia/nvidia-nemotron-nano-9b-v2` through NVIDIA's hosted
NIM API. It is declared as 9B parameters in source and metadata. The runner rejects
any configured model size at or above 10B and never routes to a larger fallback.
Each trace record also stores NIM latency and the provider-reported prompt,
completion and total token counts, so performance claims can be reproduced.

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
