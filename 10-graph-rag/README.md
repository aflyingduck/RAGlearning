# 10 — Graph RAG

Every prior project retrieved chunks by similarity — either to the query
directly (01-03) or via lexical/rerank/filter adjustments on top of that
(04-08). Graph RAG answers a different kind of question: one where the
answer isn't semantically similar to the query at all, because it's
connected to it through a chain of relationships instead.

This corpus gained two internal docs for this project: `escalation-
routing.md` (which team handles which issue type, and under which
"Support Track" number) and `support-track-slas.md` (what each Support
Track's resolution-time commitment is). The connection between them is a
shared opaque label ("Support Track 3") — deliberately chosen to carry no
semantic content of its own, the same trick 04-hybrid-search used with
error codes.

```
cd 10-graph-rag
../.venv/Scripts/python.exe flat_search.py "your question" --top-k 5   # baseline
../.venv/Scripts/python.exe graph_search.py                            # traversal
```

## The question flat search genuinely cannot answer

*"My two hubs won't mesh together and support determined it's a hardware
defect — which team handles it now and how long will it take?"*

Answering it requires three hops across two documents: Hub Mesh issue →
**Connectivity Team** triages it → reclassified as hardware fault →
escalated to **Hardware Team** → Hardware Team resolves under **Support
Track 3** → Support Track 3's SLA is **7 business days**. No single chunk
states the full chain, and the last hop (`"Support Track 3: resolved
within 7 business days"`) shares almost no vocabulary with the original
question at all.

```
../.venv/Scripts/python.exe flat_search.py "My two hubs won't mesh together and support determined it's a hardware defect -- which team handles it now and how long will it take?" --top-k 5
```

Real output, top 5:
```
[1] (0.581) escalation-routing.md: "...escalated to the Hardware Team, who resolve it under Support Track 3."
[2] (0.557) escalation-routing.md: "Networking issues ... triaged first by the Connectivity Team."
[3] (0.551) warranty-and-returns.md: "Warranty claims are typically processed within 5 business days..."
[4] (0.503) escalation-routing.md: (privacy/software routing, not relevant)
[5] (0.502) troubleshooting.md: "Hub Mesh shows a paired hub as offline..."
```

Flat search correctly finds **who** handles it (Hardware Team, [1]) — but
the actual SLA chunk from `support-track-slas.md` never appears in the top
5 at all. Worse, **[3] is a coincidentally plausible wrong answer**: "5
business days" is a real number from the corpus, about a real warranty
process, that has nothing to do with Support Track SLAs — and it looks
exactly like the kind of number this question is asking for. A system that
just handed an LLM these 5 chunks would have a genuine chance of
confidently answering "5 business days," which is wrong. This is a more
dangerous failure than an obvious miss: a plausible, present, *wrong*
number sitting right where the right one should be.

## Graph traversal gets the right chain

```
../.venv/Scripts/python.exe graph_search.py
```
```
Step 1: it starts as a networking issue --
  issue:networking --triaged_by--> team:connectivity

Step 2: triage reclassifies it as a hardware defect --
  issue:hardware_defect --escalated_to--> team:hardware
  team:hardware --resolves_under--> track:3
  track:3 --sla--> sla:7_business_days

Answer: Hardware Team, under Support Track 3, resolved within 7 business days.
```

`graph_search.py` builds a small `networkx` graph (`build_graph()`) with
nodes for issue types, teams, and support tracks, and edges for the
relations connecting them ("triaged_by," "escalated_to," "resolves_under,"
"sla"). `nx.shortest_path` then walks from the starting issue to the
answer, and every hop is explicit and inspectable — nothing is inferred
from embedding proximity.

## What this project doesn't do (and why)

Real Graph RAG systems extract the graph automatically, usually with an
LLM reading the corpus and pulling out entities and relationships. This
project hand-builds the graph in `build_graph()` instead, and hands the
starting node (`issue:hardware_defect`) to the traversal directly rather
than parsing it from the free-text question. That's a deliberate scope
cut, not laziness: entity extraction and query-to-node mapping are both
generation tasks (reading text, producing structured output), which per
this curriculum's pattern belong in an interactive step with Claude, not
hardcoded into a retrieval script. What this project isolates and
demonstrates is the piece that's actually different about Graph RAG: once
you have a graph, traversal answers a class of question — relationship
chains through low-vocabulary-overlap connector nodes — that similarity
search structurally cannot, no matter how the query is phrased, how it's
chunked, or how many results you retrieve.

## When it's worth the cost

Building and maintaining a graph is real, ongoing work — entity
extraction, keeping the graph in sync as source docs change, and (for
free-text queries) a reliable way to map a question onto starting nodes.
It's worth that cost when your domain genuinely has this shape: multi-hop
relationships through connector entities that don't share vocabulary with
how people ask about them (org charts, ownership/routing rules, ticket
escalation, regulatory chains). If your corpus is mostly self-contained
prose — like every other project's Nimbus Hub docs — plain chunking and
vector search (01-08) answer nearly everything, and a graph is
maintenance overhead for a problem you don't have.
