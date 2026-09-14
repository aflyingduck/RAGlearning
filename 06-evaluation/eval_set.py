"""
A labeled evaluation set: (query, source doc, a distinctive substring that
must appear in a retrieved chunk for it to count as relevant), grouped
into four categories chosen to actually separate the methods being
compared -- not just confirm they all work.

Substring containment -- not chunk index -- is the relevance judgment,
because chunk boundaries differ across the strategies being compared here
(fixed vs. sentence vs. recursive vs. semantic). A chunk index that's
"correct" for one chunking strategy is meaningless for another; the actual
answer text is the only thing that's comparable across all of them.

Categories:
- "easy": a clean, single fact with fairly unique vocabulary. Every method
  should get these right -- they're the floor, not the interesting part.
- "boundary": the answer text sits exactly across a real fixed-size
  chunking cut point (verified against actual chunk_fixed() output, not
  guessed -- see 06's README). 50-char overlap turns out to rescue RECALL
  for all four of these on this corpus, so the interesting signal here is
  rank/score degradation from a chunk that mixes two unrelated topics
  together, not outright misses.
- "exact_token": the answer hinges on an opaque token (an error code, a
  protocol version number) that carries little semantic meaning on its
  own -- the kind of thing 04-hybrid-search showed vector search
  struggling with and BM25 nailing instantly.
- "compound": the answer sentence is embedded inside a paragraph that also
  discusses a related-but-different aspect of the same topic, so a single
  query embedding has to represent a specific fact while sitting next to
  more generic neighboring text. This is where reranking (05) earns its
  keep -- see 06's README for the specific rank movements.
"""

EVAL_SET = [
    # -- easy --
    {
        "category": "easy",
        "query": "How long is the warranty on the Hub 3?",
        "doc": "warranty-and-returns.md",
        "substring": "2-year limited hardware warranty",
    },
    {
        "category": "easy",
        "query": "What does the warranty not cover?",
        "doc": "warranty-and-returns.md",
        "substring": "does not cover physical damage",
    },
    {
        "category": "easy",
        "query": "Which older Hub model isn't sold anymore?",
        "doc": "product-overview.md",
        "substring": "still supported but no longer sold",
    },
    {
        "category": "easy",
        "query": "How much power does the Hub use at idle?",
        "doc": "product-overview.md",
        "substring": "6 watts at idle",
    },
    {
        "category": "easy",
        "query": "What should I do if a device shows online but won't respond?",
        "doc": "troubleshooting.md",
        "substring": "remove and re-pair it",
    },
    {
        "category": "easy",
        "query": "What Wi-Fi band does the Hub need for initial setup?",
        "doc": "installation-guide.md",
        "substring": "only supports 2.4 GHz Wi-Fi for initial setup",
    },
    # -- boundary: substring verified to straddle a real chunk_fixed() cut point --
    {
        "category": "boundary",
        "query": "How do I file a warranty claim?",
        "doc": "warranty-and-returns.md",
        "substring": "solsticelabs.com/support",
    },
    {
        "category": "boundary",
        "query": "Is my voice data ever shared with other Solstice Labs customers?",
        "doc": "privacy-and-security.md",
        "substring": "never used to train models",
    },
    {
        "category": "boundary",
        "query": "Does a lapsed Nimbus Plus subscription affect how fast my automations run?",
        "doc": "subscription-plans.md",
        "substring": "does not change local automation speed",
    },
    {
        "category": "boundary",
        "query": "How do I factory reset the Hub?",
        "doc": "troubleshooting.md",
        "substring": "flashes purple",
    },
    # -- exact_token: opaque codes/identifiers, low semantic content --
    {
        "category": "exact_token",
        "query": "What does error ERR-4471 mean?",
        "doc": "troubleshooting.md",
        "substring": "firmware signature verification failed",
    },
    {
        "category": "exact_token",
        "query": "What does error ERR-2208 mean?",
        "doc": "troubleshooting.md",
        "substring": "Zigbee radio failed to initialize",
    },
    {
        "category": "exact_token",
        "query": "ERR-1090",
        "doc": "troubleshooting.md",
        "substring": "Hub Mesh link rejected",
    },
    {
        "category": "exact_token",
        "query": "What version of TLS encrypts Hub-to-cloud traffic?",
        "doc": "privacy-and-security.md",
        "substring": "TLS 1.3",
    },
    # -- compound: the answer sits next to related-but-different content --
    {
        "category": "compound",
        "query": "Do I need a subscription for local automations to work?",
        "doc": "subscription-plans.md",
        "substring": "works fully offline for local automations without a subscription",
    },
    {
        "category": "compound",
        "query": "What happens to my automations if I stop paying for Nimbus Plus?",
        "doc": "subscription-plans.md",
        "substring": "keeps functioning locally",
    },
    {
        "category": "compound",
        "query": "I have a huge house and one of these little boxes probably can't cover it all -- what are my options?",
        "doc": "product-overview.md",
        "substring": "up to 200 connected devices per hub",
    },
    {
        "category": "compound",
        "query": "Will Solstice Labs replace my Hub before I send the broken one back?",
        "doc": "warranty-and-returns.md",
        "substring": "ship a replacement before requiring the",
    },
]
