"""
Graph RAG: instead of searching flat chunks, extract entities and
relationships into a graph and answer by TRAVERSING it.

flat_search.py showed a real failure: the query and the fact that answers
it ("Support Track 3: resolved within 7 business days") share almost no
vocabulary, because the connection between them is a relationship
(HardwareDefect --escalated_to--> Hardware --resolves_under--> Track3 --sla-->
7 business days), not a topic. Cosine similarity over independent
embeddings has no way to represent "these two facts are two hops apart in
a chain" -- a graph makes that chain an explicit, followable structure.

The entity extraction here is hand-built (see build_graph()) rather than
LLM-extracted from the docs -- real Graph RAG systems usually use an LLM
to pull entities/relations out of text automatically. That's a
generation step and, per this curriculum's pattern, would happen
interactively rather than via an API call baked into this script; this
project focuses on what traversal buys you once a graph exists, not on
automating its construction.

Usage:
    python graph_search.py
"""

import sys

import networkx as nx

sys.stdout.reconfigure(encoding="utf-8")


def build_graph():
    """Hand-extracted from escalation-routing.md and support-track-slas.md."""
    g = nx.DiGraph()

    g.add_edge("issue:networking", "team:connectivity", relation="triaged_by")
    g.add_edge("issue:hardware_defect", "team:hardware", relation="escalated_to")
    g.add_edge("issue:billing", "team:billing", relation="handled_by")
    g.add_edge("issue:privacy", "team:trust_safety", relation="routed_to")
    g.add_edge("issue:software_firmware", "team:connectivity", relation="stays_with")

    g.add_edge("team:hardware", "track:3", relation="resolves_under")
    g.add_edge("team:billing", "track:1", relation="resolves_under")
    g.add_edge("team:trust_safety", "track:4", relation="resolves_under")
    g.add_edge("team:connectivity", "track:2", relation="resolves_under (software/firmware only)")

    g.add_edge("track:1", "sla:1_business_day", relation="sla")
    g.add_edge("track:2", "sla:3_business_days", relation="sla")
    g.add_edge("track:3", "sla:7_business_days", relation="sla")
    g.add_edge("track:4", "sla:14_calendar_days", relation="sla")

    return g


def explain_path(g, start, end):
    path = nx.shortest_path(g, start, end)
    steps = []
    for a, b in zip(path, path[1:]):
        steps.append(f"{a} --{g.edges[a, b]['relation']}--> {b}")
    return path, steps


def main():
    g = build_graph()

    print("Scenario: a Hub Mesh (networking) issue triaged, then found to be a hardware defect.\n")

    # A real system would use an LLM (or a classifier) to map free text to
    # these starting nodes; here the starting nodes are given directly to
    # isolate what TRAVERSAL adds, separate from entity extraction.
    print("Step 1: it starts as a networking issue --")
    path, steps = explain_path(g, "issue:networking", "team:connectivity")
    for s in steps:
        print(f"  {s}")

    print("\nStep 2: triage reclassifies it as a hardware defect --")
    path, steps = explain_path(g, "issue:hardware_defect", "sla:7_business_days")
    for s in steps:
        print(f"  {s}")

    print(f"\nAnswer: Hardware Team, under Support Track 3, resolved within 7 business days.")
    print("This chains three separate facts from two different documents that share almost")
    print("no vocabulary with the original question or each other -- see README for why")
    print("flat_search.py cannot find the SLA step of this chain at all.")


if __name__ == "__main__":
    main()
