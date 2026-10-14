"""Shadow a keyword classifier with Jev on a few tickets, then print the report.

    OPENROUTER_API_KEY=... python examples/ticket_routing.py
"""

from decision_shadow import Shadow, choice
from decision_shadow.cli import main

TICKETS = [
    "I was charged twice for September. Please refund the duplicate charge.",
    "The CSV export returns a 500 error for every workspace.",
    "How do I add three more seats to our plan?",
    "Our invoice shows the wrong company address.",
    "I can't log in since I changed my email.",
    "The dashboard is blank in Safari but fine in Chrome.",
]

shadow = Shadow(log="decisions.jsonl", mode="shadow")


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
    decide=lambda d: d["team"].choice,
)
def route_ticket(ticket: str) -> str:
    t = ticket.lower()
    if any(w in t for w in ("charge", "refund", "invoice")):
        return "billing"
    if any(w in t for w in ("log in", "seat", "email")):
        return "account"
    return "technical"


if __name__ == "__main__":
    for t in TICKETS:
        print(f"{route_ticket(t):<10} {t}")
    shadow.flush()
    main(["report", "decisions.jsonl"])
