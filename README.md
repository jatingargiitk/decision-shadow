# decision-shadow

Swap an LLM classifier for a decision model like [TypeSafe Jev](https://openrouter.ai/docs/guides/community/jev) without trusting it blind.

You already have a function that asks a chat model "is this billing or technical?" and parses the reply. Jev answers the same question with typed choices and probabilities, in a few hundred milliseconds and for a fraction of a cent. The open question is whether it answers *your* question as well as the code it replaces. `decision-shadow` finds that out on your real traffic:

1. **shadow** — your function keeps deciding. The model runs alongside it, concurrently, and both answers go to a JSONL log.
2. **report** — per decision site: agreement, latency, cost, the most common disagreements, and warnings when probabilities are too saturated to threshold on or the model snapshot changed underneath you.
3. **tune** — sweep certainty thresholds and pick the lowest one that hits your accuracy target, against labels if you have them.
4. **on** — the model decides; anything below your threshold, and every error, falls back to your function.

No dependencies. Python 3.9+.

```bash
pip install git+https://github.com/jatingargiitk/decision-shadow
export OPENROUTER_API_KEY=...
```

## Wrap a classifier

```python
from decision_shadow import Shadow, choice

shadow = Shadow(log="decisions.jsonl")  # mode from $DECISION_SHADOW_MODE: off | shadow | on

@shadow.site(
    "ticket-routing",
    state=lambda ticket: {"ticket": ticket},
    questions={
        "team": choice("Which team owns this support ticket?", {
            "billing": "Charges, refunds, invoices",
            "technical": "Bugs, errors, broken features",
            "account": "Login, seats, profile changes",
        }),
    },
    decide=lambda d: d["team"].choice if d["team"].certainty >= 0.8 else None,
)
def route_ticket(ticket: str) -> str:
    return my_llm_classifier(ticket)  # the code you have today
```

`decide` maps the model's answers onto whatever your function returns. Returning `None` means "not sure" — in `on` mode that runs your function instead. Shadow mode never changes what your function returns and adds no latency: the model call starts before your function and is logged when it lands.

State isn't logged by default, only a hash of it. Pass `log_state=True` if you want it in the log for review.

## Read the log

This is the real output of [`examples/ticket_routing.py`](examples/ticket_routing.py) — six tickets, a keyword classifier as the "legacy" side, `typesafe/jev-1.13` as the model:

```text
$ decision-shadow report decisions.jsonl

ticket-routing
  decisions   6  (errors 0, fallbacks 0)
  agreement   100.0% of 6 compared
  latency     model p50 494 ms / p95 575 ms   legacy p50 0 ms
  cost        $0.000089
  saturation  66.7% of 6 answers at >= 0.99
  ! only 6 compared decisions — too few to trust the agreement rate
```

When the two disagree, the report lists the most common `legacy -> model` pairs, which is usually where the question wording needs work.

```bash
decision-shadow tune decisions.jsonl --site ticket-routing --question team --target 0.97
```

prints, for each certainty threshold, the share of traffic above it (coverage), the accuracy on that share, and the lowest threshold that reaches the target on at least `--min-n` decisions.

Without `--labels`, "accuracy" is agreement with the classifier being replaced — useful for spotting drift, not proof the model is right. Label a few hundred decisions (`{"id": "<record id>", "label": "billing"}` per line) and pass `--labels labels.jsonl` to measure against ground truth.

## Things the report warns about

- **Saturated probabilities.** If most answers come back at 0 or 1, a confidence threshold can't separate easy cases from hard ones, and the threshold sweep tells you little. This happens with Jev on clear-cut inputs; check it on your own data before relying on "calibrated" thresholds.
- **Snapshot changes.** `~typesafe/jev-latest` moves. The log records the dated snapshot that served each call; if a site's log spans two, re-tune. Pin `typesafe/jev-1.13` for stable thresholds.
- **Small samples.** Fewer than 30 compared decisions and the agreement rate is noise.

## Questions

`choice`, `noul` (yes/no), and `score` (ordinal, up to 10 levels) build the three question types the Decisions API accepts. All questions in one request see the same state and are answered in one call.

```python
from decision_shadow import choice, noul, score

noul("Is the customer asking for money back?", true="Asks for a refund", false="Does not ask for money")
score("How severe is this bug?", ["Cosmetic", "Degraded, workaround exists", "Blocking"])
```

Every answer has `.value` (label, weighted level, or P(yes)) and `.certainty` — one number in [0, 1] to threshold on regardless of type.

## Backends

`JevBackend` talks to OpenRouter's Decisions API by default, or TypeSafe's own endpoint:

```python
from decision_shadow import JevBackend, Shadow
from decision_shadow.backends import TYPESAFE_URL

Shadow(backend=JevBackend(model="typesafe/jev-1.13"))          # OpenRouter, pinned
Shadow(backend=JevBackend(url=TYPESAFE_URL))                    # uses $TYPESAFE_API_KEY
```

Any object with `decide(state, questions) -> Decision` works as a backend, so a local Jev-compatible model plugs in the same way.

## License

MIT
