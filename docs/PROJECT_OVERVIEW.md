# Agent Memory Challenge: Project Overview

This document explains what the project is, why it exists, how every part
works, and where it stands. It is written for someone new to the repository.
For the short version and the commands, see `README.md`.

**Contents**

1. [What this project is](#1-what-this-project-is)
2. [The problem: how agent memory goes wrong](#2-the-problem-how-agent-memory-goes-wrong)
3. [How the benchmark works, end to end](#3-how-the-benchmark-works-end-to-end)
4. [Repository layout](#4-repository-layout)
5. [The memory-system interface](#5-the-memory-system-interface-interfacememory_systempy)
6. [Planted-fact timelines: the ground truth](#6-planted-fact-timelines-the-ground-truth-workloadstimelinepy)
7. [Workloads: the hand-built example and the generator](#7-workloads-the-hand-built-example-and-the-generator)
8. [The runner: replaying a timeline](#8-the-runner-replaying-a-timeline-runnerreplaypy)
9. [Scoring](#9-scoring-runnerscoringpy)
10. [The systems under test](#10-the-systems-under-test)
11. [The ledger, in depth](#11-the-ledger-in-depth-systemsledger)
12. [LLM providers and configuration](#12-llm-providers-and-configuration)
13. [Results so far](#13-results-so-far)
14. [Running it](#14-running-it)
15. [Tests](#15-tests)
16. [Known limitations and next steps](#16-known-limitations-and-next-steps)
17. [Glossary](#17-glossary)

---

## 1. What this project is

A **benchmark for long-term memory in AI agents**, focused on coding agents.

An agent that works with a user over weeks (on their services, repositories,
infrastructure) has to remember things: which database a service runs on, how
to run the tests, where staging lives. A *memory system* is the component that
stores what the agent has seen and hands relevant pieces back when a question
comes up.

The project does three things:

1. **Defines a common interface** (`MemorySystem`) that any memory system can
   implement: simple baselines, third-party systems (Mem0, A-MEM, ...), and
   our own system.
2. **Builds test workloads with planted ground truth.** Every workload is a
   stream of agent interactions in which known facts are introduced, changed,
   contradicted and deleted at known times. Because we planted the facts, we
   know the correct answer to every question at every moment, and we know
   *why* any wrong answer is wrong.
3. **Replays workloads against systems and scores them**, with the same rules,
   token budgets and accounting for every system, so the comparison is fair.

It also contains **our own memory system, the ledger**: a fact store that keeps
the history of each fact, ranks sources by trust, deletes properly, and
declines to answer when it does not know.

## 2. The problem: how agent memory goes wrong

Simply keeping the whole history, or searching over it, fails in predictable
ways. The benchmark is built around seven properties, called invariants
I1–I7 in the code:

| Invariant | Property | Failure it catches | Example |
|---|---|---|---|
| **I1** deletion | A fact the user asked to forget is never returned again, including in questions about the past. | *Resurrection*: the deleted value comes back. | "Forget the staging hostname" → later asked for it, the system still shows `staging-7.internal`. |
| **I2** supersession | When a fact changes, the old value stops being the answer. | *Stale* values. | Database moved from PostgreSQL 14 to 16; the system still shows 14. |
| **I3** temporal | "What was true at time *t*?" returns the value from then, not a later one. | *Future* values. | "Which database at the end of January?" answered with February's migration. |
| **I4** deduplication | Re-ingesting the same event, or restating the same fact, does not create duplicate memory. | Store bloat, duplicate items. | "Reminder: still on Postgres 14" should not add a second item. |
| **I5** contamination | A less trustworthy source (an outdated README, the agent's own guess) does not override what the user said. | *Contaminated* values. | User says deploy from `main`; an old README says `master`; the system answers `master`. |
| **I6** abstention | When nothing is known, the system says so instead of returning something loosely related. | Confident noise. | "Which message queue?" when none was ever mentioned. |
| **I7** isolation | One user's memory never appears in another user's answers. | Cross-user leaks. | User B sees user A's database. |

The code also refers to three requirements (R1–R3) and three challenge areas
(C1–C3). As used in code comments:

- **R1**: a capability report per system: provenance, temporal queries,
  deletion, abstention, persistence, export (claimed vs. verified).
- **R2**: a fair comparison: same token budgets, one tokenizer, one price
  table; latency and cost are measured by the runner, never self-reported.
- **R3**: reproducibility: every call and result is logged as JSON so a run
  can be audited and re-scored.
- **C1**: heterogeneous traces (messages, tool calls, documents, code changes)
  and supersession. **C2**: questions about the past. **C3**: isolation
  between users or agents.

## 3. How the benchmark works, end to end

```
 workload (Timeline JSON)                     memory system (MemorySystem)
 ┌──────────────────────────┐                ┌──────────────────────────┐
 │ events      (the stream) │── ingest ─────▶│ store what matters       │
 │ facts + mutations (truth)│── delete ─────▶│ forget on request        │
 │ checkpoints              │                │                          │
 │ probes (questions)       │── retrieve ───▶│ return evidence, or      │
 └──────────────────────────┘                │ abstain                  │
              │                              └────────────┬─────────────┘
              │ expected answers                          │ evidence
              ▼                                           ▼
        ┌──────────────────────────────────────────────────────┐
        │ runner: truncate to the token budget, time each call,│
        │ score the evidence against ground truth, log as JSON │
        └──────────────────────────────────────────────────────┘
                                   │
                                   ▼
                 report table + runs/<workload>/<system>.json
```

1. A **timeline** is loaded: events, the facts planted in them, how those
   facts change (mutations), checkpoints and probes.
2. The **runner** feeds events to the system one session at a time, delivers
   deletion requests when the user makes them, and at each checkpoint asks
   the probes.
3. The system answers each probe with **evidence** (text snippets it would
   put into the agent's context) or **abstains**.
4. The runner cuts evidence to the **token budget**, then **scores** it: did
   the correct value appear, did any forbidden value appear, and if so, why
   was it forbidden (stale, future, contaminated, resurrected)?
5. Everything is **logged** to `runs/<workload>/<system>.json` and summarized
   in a comparison table.

The benchmark scores the *evidence*, not a generated answer. That isolates the
memory system from the answering model: if the right value is not in the
evidence, no model can answer correctly; if a wrong value is there next to the
right one, the answer is at risk.

## 4. Repository layout

```
interface/
  memory_system.py     The MemorySystem API and all data types (Event, Query, Evidence, ...)
  tokens.py            The one tokenizer used to count evidence tokens
workloads/
  timeline.py          Timeline schema: facts, mutations, probes, ground-truth derivation, validation
  examples/
    build_example.py   Hand-built 12-event timeline (smoke test for every feature)
    example_timeline.json
  generator.py         Seeded synthetic timeline generator (presets small, medium)
  generated/           Generated timelines (e.g. synthetic-small-s0.json)
runner/
  replay.py            Replays a timeline against one system; logs everything
  scoring.py           Verdicts and error types
  __main__.py          CLI: run systems, save logs, print the comparison table
systems/
  no_memory/           Lower bound: stores nothing
  transcript_store.py  Shared storage for the two transcript baselines
  full_replay/         Returns the whole history (recall upper bound)
  transcript_rag/      BM25 search over raw events
  ledger/              Our system
    system.py          The ledger core (history, trust, deletion, retrieval)
    extract.py         Extractor interface + rule-based extractor
    llm.py             LLM-based extractor (any provider), with response cache
    backends.py        Model providers: Anthropic, OpenAI / OpenAI-compatible
runs/                  Run logs, one JSON file per (workload, system, mode)
tests/                 67 tests (schema, runner, ledger, generator, LLM extractor)
.env.example           Template for API keys and model settings (.env is git-ignored)
```

## 5. The memory-system interface (`interface/memory_system.py`)

Every system implements `MemorySystem`. The runner talks to systems only
through it, which is what makes comparisons fair.

### Required methods

| Method | Purpose |
|---|---|
| `ingest(events) -> WriteReport` | Store one session's events, in order. Re-ingesting a seen event must not create new items (I4). |
| `retrieve(query) -> RetrievalResult` | Return evidence for a question within the token budget, or abstain. Must use only `query.user_id`'s memory (I7). |
| `stats(user_id) -> StoreStats` | Store size, for storage-growth curves. |
| `reset()` | Erase everything between runs. |
| `config() -> dict` | Every setting that affects behaviour (models, thresholds, prompt hashes), logged with each run. |

### Optional methods

Each one is declared in `Capabilities`; unsupported ones raise `NotSupported`,
which the runner records as "unsupported" instead of crashing.

| Method | Purpose |
|---|---|
| `delete(request)` | Forget a fact (I1). |
| `snapshot(path)` / `restore(path)` | Persist and reload the store. |
| `export_items(user_id)` | White-box dump of stored items, for audits. |

### Data types

- **`Event`**: one item of an interaction trace: `event_id` (globally unique),
  `user_id`, `session_id`, `timestamp`, `kind`, `content`. Kinds:
  `user_message`, `assistant_message`, `tool_call`, `tool_result`, `document`,
  `code_change`.
- **`Query`**: `text`, `user_id`, `asked_at` (when it is asked), optional
  `as_of` (for "what was true then" questions), `token_budget` (default 2000;
  `None` = unbounded, only for full replay).
- **`Evidence`**: one retrieved item: `content` plus optional provenance
  (`source_event_ids`), validity interval (`valid_from`, `valid_to`),
  `recorded_at`, `score`.
- **`RetrievalResult`**: evidence, or `abstained=True` with a reason. An
  abstaining result may not carry evidence (enforced by validation).
- **`DeleteRequest`**: `description` (the user's words, e.g. "Please forget the
  staging hostname") and optionally `source_event_ids` (the events that stated
  the fact). Experiments can withhold the sources to test deletion by
  description only.
- **`Usage`**: LLM calls, input/output tokens, embedding tokens, model ids.
  Systems report what they consume; the runner converts to cost.
- **`Capabilities`**: what a system *claims*: provenance, temporal_query,
  deletion, abstention, persistence, export.

### Design rules

1. Black-box first: the runner needs only `ingest`, `retrieve`, `delete`, `stats`.
2. Everything is scoped by `user_id`.
3. Systems report token usage; the **runner** measures latency, counts
   injected tokens with one tokenizer (`interface/tokens.py`, a word/punctuation
   counter, version `regex-words/1`), and owns cost. No system can flatter itself.
4. All types are pydantic models, so every call can be logged and re-scored.

## 6. Planted-fact timelines: the ground truth (`workloads/timeline.py`)

A `Timeline` bundles:

- **events**: the interaction stream the system sees.
- **facts**: slots whose values the stream sets, each identified by
  `(user_id, entity, attribute)`, e.g. `(u1, payments-service, database)`.
- **mutations**: how each fact changes over time, each tied to the event(s)
  that express it (`carrier_event_ids`).
- **checkpoints**: points in the stream where questions are asked.
- **probes**: the questions, each with its expected answer.

### Mutation kinds

| Kind | Meaning | Effect on the truth |
|---|---|---|
| `INTRODUCE` | Fact first stated. | Truth becomes the value. |
| `UPDATE` | The world changed. | Truth becomes the new value; the old one is superseded. |
| `RESTATE` | Same value, said again (possibly worded differently). | No change. Tests deduplication. |
| `CONTRADICT` | A less trusted source asserts a conflicting value. | No change. Tests contamination. |
| `DELETE` | The user asks to forget the fact. | Every earlier value becomes unretrievable, including for questions about the past. A later `INTRODUCE` starts a fresh history. |

### Sources and trust

A mutation's source is fixed by the kind of event that carries it, so a real
system could know it from the trace alone:

| Event kind | Source |
|---|---|
| `user_message` | user |
| `tool_result`, `code_change` | tool |
| `document` | document |
| `assistant_message`, `tool_call` | model |

Rules enforced on mutations: a `CONTRADICT` never comes from the user (a user
stating a new value is an `UPDATE`); a `DELETE` always comes from the user.

### Probe types

| Probe type | Question | Expected | Forbidden values typed as |
|---|---|---|---|
| `current` | Plain question about a known fact | The value | n/a |
| `post_update` | Asked after an update | The new value | stale |
| `historical` | Asked with `as_of` in the past | The value at `as_of` | future |
| `post_delete` | Asked after a deletion | Abstain | resurrected |
| `contamination` | Asked after a contradiction | The trusted value | contaminated |
| `abstention` | About something never stated | Abstain | n/a |
| `multi_fact` | Combines several facts (hand-written) | Hand-written | forbidden |

### Ground truth is derived, never hand-typed

`expected_answer(mutations, asked_at, as_of)` computes each probe's answer
from the mutations: find the last deletion before `asked_at`; everything
before it is erased (forbidden as *resurrected*); among the rest, the last
`INTRODUCE`/`UPDATE` at or before `as_of` (or `asked_at`) is the truth; every
other value the fact ever had (including contradictions) is forbidden.

### Validation on load

`Timeline` refuses to load if anything is inconsistent. It checks:

- Unique ids.
- One fact per slot.
- Events in chronological order.
- Each mutation's time equals its carrier event's time, and its source matches
  the carrier's kind.
- Carriers belong to the fact's user.
- Each fact's mutations form a legal sequence (no `UPDATE` to the same value,
  no `RESTATE` of a different value, no `CONTRADICT` of an inactive fact, ...).
- Probes refer to real facts of the same user.
- Probe types are backed by the right mutations (e.g. a `post_update` probe
  needs an `UPDATE` before its checkpoint).
- **Every single-fact probe's expected answer is recomputed** from the
  mutations and must match.

A buggy generator or a hand edit therefore cannot silently corrupt results.

## 7. Workloads: the hand-built example and the generator

### The hand-built example (`workloads/examples/build_example.py`)

Twelve events for one user across five sessions, exercising every mutation and
probe type. It is the smoke test for adapters and the runner.

| Event | Time | Kind | Content | Mutation |
|---|---|---|---|---|
| e01 | Jan 5 | user | For payments-service we're on PostgreSQL 14. | introduce database |
| e02 | Jan 5 | user | Run the tests with `pytest -x`, please. | introduce test command |
| e03 | Jan 5 | user | We deploy from the main branch. | introduce deploy branch |
| e04 | Jan 5 | assistant | Got it, I'll keep that in mind. | |
| e05 | Jan 20 | document | README.md: Deploys go out from the `master` branch. | contradict (stale doc) |
| e06 | Jan 20 | user | Staging lives at staging-7.internal. | introduce staging host |
| e07 | Jan 20 | user | Reminder: payments still runs on Postgres 14. | restate |
| e08 | Feb 15 | tool result | migrate.log: payments-service migrated to PostgreSQL 16. OK | update database |
| e09 | Feb 15 | user | Tests now run with `make test`. | update test command |
| e10 | Mar 10 | user | Please forget the staging hostname, it shouldn't be stored. | delete staging host |
| e11 | Mar 10 | assistant | Understood. Deploying from master as usual, then? | contradict (agent echo) |
| e12 | Mar 30 | user | Let's refactor the cart module next. | |

Three checkpoints (after e07, e09, e12) and eight probes: current database,
deploy branch twice (contamination), database and test command after updates,
the database "at the end of January" (historical), the staging host after
deletion, and a never-mentioned message queue (abstention).

### The generator (`workloads/generator.py`)

Produces larger, varied, reproducible timelines from a seed.

**What it generates**

- **Users and services.** Each user owns several services, drawn from
  `payments-service`, `cart-service`, `auth-service`, and others.
- **Fact kinds.** Nine kinds, each with a value pool:

  | Kind | Example values |
  |---|---|
  | database | PostgreSQL 12–17, MySQL, MariaDB, MongoDB |
  | test command | `pytest -x`, `make test`, `go test ./...`, ... |
  | deploy branch | main, master, trunk, ... |
  | staging hostname | `payments-stg-3.internal`, ... |
  | runtime | Python 3.12, Node 20, Go 1.22, Java 21, ... |
  | message queue | RabbitMQ, Kafka, Amazon SQS, ... |
  | CI provider | GitHub Actions, GitLab CI, ... |
  | cloud region | eu-north-1, us-east-1, ... |
  | on-call owner | person names |

  Each service gets a random subset, so some kinds are missing. That is where
  abstention questions come from.
- **Mutation sequences.** A state machine generates each fact's history:
  `INTRODUCE`, then a random mix of restate, update, contradict and delete,
  with a fresh introduce after a deletion. Private data (hostnames, on-call
  owners) is deleted more often.
- **Phrasing.** Every mutation is written by a randomly chosen template for its
  source (user message, tool output, code diff, document, assistant), mixing
  generic templates with kind-specific ones. The variety is deliberate: an
  extractor tuned to the hand-built example should not cover it all.
- **Sessions and timing.** Each user's sessions fall on distinct days over a
  span (180 days for `small`). A fact's mutations go to increasing sessions,
  so they are strictly ordered in time.
- **Filler and distractor events.** Chit-chat, tool calls and messages that
  name a topic without a value ("The runtime for cart-service came up in
  standup today").
- **Probes.** At each checkpoint, for each user and each probe type, the
  generator samples up to `probes_per_type` facts whose state makes that type
  meaningful.

**Guarantees** (so the scoring stays unambiguous)

- A value belongs to exactly one fact per user, and no value is a sub-phrase of
  another, so "value X appeared" always points to one fact.
- No event mentions a value it does not carry, and no question mentions any
  value. `hygiene_errors` checks this after every generation and raises if
  violated.
- A `CONTRADICT` comes only from a document or the assistant, and only against
  a value the user or a tool established. A stale document contradicting a
  value that itself came from a document would be a judgement call, not
  ground truth.
- Every timeline passes the schema's full validation.

**Presets**

| | small | medium |
|---|---|---|
| Users | 3 | 8 |
| Services per user | 2–3 | 2–4 |
| Fact kinds per service | 3–5 | 3–5 |
| Sessions per user | 16 | 40 |
| Mutations per fact | 1–5 | 1–8 |
| Span | 180 days | 365 days |
| Checkpoints | at 25 / 50 / 75 / 100 % of events | same |
| Probes per user, checkpoint and type | up to 2 | up to 2 |
| Seed 0 output | 216 events, 31 facts, 89 mutations, 99 probes | 1,211 events, 391 mutations, 306 probes |

`multi_fact` probes are not generated (they need hand-written expectations).

## 8. The runner: replaying a timeline (`runner/replay.py`)

### Protocol (the contract systems can rely on)

- **Ingest.** Events arrive in timeline order. A session's events go to
  `ingest` in one call once its last event has arrived. A checkpoint or a
  deletion in the middle of a session flushes what has arrived so far; the
  rest follows in a later call.
- **Deletes.** Each `DELETE` mutation becomes a `DeleteRequest`, delivered right
  after the user's "please forget..." event has been ingested. Its
  `source_event_ids` are the events that stated the fact before the deletion.
  With `--withhold-sources` they are left out, so the system must work from
  the description alone.
- **Probes.** At each checkpoint, every event up to that time has been
  ingested. Each probe becomes a `Query` with `asked_at` = checkpoint time. A
  system never sees an event from after the question.
- **Budget.** The runner counts each evidence item's tokens and keeps whole
  items, in the system's order, until the budget is reached. Overflow counts as
  a budget violation. Scoring sees only what survives.
- **Accounting.** The runner times every call itself and records the `Usage`
  the system reports.

### What gets logged

One `RunResult` JSON file per run, containing:

- Workload id, generator and seed.
- System name, version, full `config()` and `capabilities`.
- Settings: token budget, deletion mode, tokenizer.
- Every write operation (`ingest`/`delete`): stream time, event ids, latency,
  usage, report, error (e.g. "unsupported").
- Every probe: question, expected answer, evidence after truncation, tokens
  before and after truncation, abstention and reason, verdict, matched
  surface form, forbidden hits with their error types and the events they
  came from, cross-user hits, latency, usage.
- Store size after each checkpoint.

`RunResult.summary()` aggregates: correct count, verdict and error-type
counts, budget and isolation violations, unsupported operations, injected
tokens, total LLM usage, ingest time, mean retrieval latency, final store size.

### The CLI (`runner/__main__.py`)

Runs the selected systems on one timeline and saves
`runs/<workload>/<system>[__nosrc].json`. It prints a probe-by-probe table
plus the summary rows. Full replay always runs with an unbounded budget.
Systems that cost money (`ledger_llm`) run only when named with `--systems`.

## 9. Scoring (`runner/scoring.py`)

A value **surfaces** when any evidence item mentions it, or one of its
accepted spellings (aliases such as "Postgres 14" for "PostgreSQL 14"). The
match is case-insensitive and whole-phrase, so `main` does not match
`maintain`.

### Verdicts

| Verdict | When |
|---|---|
| `correct` | The expected value surfaced and nothing forbidden did; or the system abstained when it should. |
| `polluted` | The expected value surfaced, but so did a forbidden one. |
| `wrong` | Only forbidden values surfaced. |
| `missed` | Neither the expected nor a forbidden value surfaced. |
| `false_abstention` | The system abstained although the fact is known. |
| `leaked` | It should abstain, and a forbidden value surfaced (e.g. a deleted one). |
| `no_abstention` | It should abstain, did not, but nothing forbidden surfaced. |

### Error types

Each forbidden value that surfaced is labelled with why it is wrong, derived
from the mutations:

| Error type | Meaning |
|---|---|
| `stale` | Superseded by a later update (I2). |
| `future` | Set after `as_of` (I3). |
| `contaminated` | Asserted by a less trusted contradiction (I5). |
| `resurrected` | Stated before a deletion (I1). |
| `forbidden` | Listed by a hand-written probe but not explained by mutations. |

**Isolation (I7)** is checked separately: for each user, any value belonging
only to *other* users' facts that appears in evidence is a cross-user hit.

## 10. The systems under test

| System | Idea | Strengths | Weaknesses |
|---|---|---|---|
| `no_memory` | Stores nothing, never abstains. | Lower bound; smallest example adapter. | Knows nothing. |
| `full_replay` | Returns the user's entire history, oldest first, unbounded. | Recall upper bound: the right value is always there. | Every old, contradicted and deleted-by-description value is there too; no abstention; huge token cost. |
| `transcript_rag` | BM25 keyword search over individual events, top 5 within budget. | Cheap, deterministic, no model. | Retrieves by word overlap, so old and new values come together; cannot abstain; cannot tell sources apart. |
| `ledger` | Our fact store, with the rule-based extractor. | Every invariant built in; no model needed. | The extractor only knows four hand-written patterns. |
| `ledger_llm` | The same ledger, with the LLM extractor. | Understands free-form text; any provider. | Costs money and latency per call. |

The two transcript baselines share `systems/transcript_store.py`:

- Events are stored verbatim, keyed by id, so re-ingesting is a no-op.
- Deletion removes the listed source events and blocks them from coming back.
- Without source ids they cannot delete anything, since they have no way to
  act on a description.

## 11. The ledger, in depth (`systems/ledger/`)

The ledger separates **understanding text** (the extractor) from **keeping
facts correctly** (the core). The core never reads free text.

### Data model (`system.py`)

Memory is partitioned by user. Each user has:

- **Entries**: one per value a slot (fact) has held: slot key, value, label,
  trust, source kind, `valid_from`, `valid_to` (empty while current),
  `source_event_ids`.
- **Disputes**: conflicting values from less trusted sources, kept for audit,
  never returned.
- **Tombstones**: a record that a slot was deleted (slot, time, request), with
  no value.
- **Blocked events**: event ids that may never be ingested again for this user.

Validity times come from event timestamps: an entry is valid from the event
that set it until the event that replaced it.

### Trust ranking

| Source | Trust |
|---|---|
| User message, tool result, code change | 3 |
| Document | 2 |
| Assistant message, tool call | 1 |

A source may replace a value only if it is at least as trusted as the source
that set it. Tool output counts as much as the user, because it observes the
world directly (e.g. a migration log).

### Write path: four rules per extracted claim, in stream order

| Rule | Condition | Action |
|---|---|---|
| **introduce** | No current value. | Open an entry. |
| **restate** | Same value as current. | Add the event to its sources; no new item (I4). |
| **supersede** | Different value, source at least as trusted. | Close the current entry at this time, open a new one (I2). The old one stays queryable by `as_of` (I3). |
| **dispute** | Different value, less trusted source. | Store as a dispute; never returned (I5). |

Re-ingesting a seen event is skipped; a blocked event is rejected.

### Deletion (I1)

1. Find the affected slots: those whose entries came from the request's source
   events, or, if none or no sources were given, the slots the extractor
   matches to the description.
2. Erase every entry and dispute of those slots recorded up to the request,
   past values included.
3. Leave a value-free tombstone.
4. Block all their source events so a replay cannot bring them back.

A later statement of the fact starts a fresh, visible history.

### Read path

1. The extractor maps the question to known slots. No match → abstain ("no
   known slot matches") (I6).
2. For each slot, return the one entry valid at `as_of` (or now) and recorded
   by `asked_at`.
3. If none: abstain, with "deleted at the user's request" or "no value held at
   <date>".
4. Evidence is one line per slot, e.g.
   `database: PostgreSQL 16 (since 2026-02-15; source: tool_result)`, carrying
   provenance and validity fields. It contains only the answer, never the old
   or rejected values, which is why it is so small (about 100 tokens for 8
   probes on the example).

### Also implemented

- `stats` (with active entries, disputes and tombstones).
- `export_items` (entries marked active or superseded).
- `snapshot`/`restore` to JSON.
- Isolation by user throughout.

### Extractors (`extract.py`, `llm.py`)

The core calls the `Extractor` interface:

- `claims(events, known)`: from a session batch, return `Claim`s (event id,
  slot key, value, label).
- `slots_for(text, known)`: which known slots a question or deletion
  description is about.

`known` is the user's slot list with descriptions and current values, so an
extractor can reuse the same keys and spellings across sessions.

- **`RuleExtractor`**: deterministic, no model. Four hand-written slot rules
  (database, test command, deploy branch, staging hostname), each a value
  regex plus cue words for questions. Slot keys are attribute names only, so
  it assumes one project per user. A stand-in, not a real solution.
- **`LLMExtractor`**: two structured-output calls.
  - **Extraction:** one call per session batch, asking for every assertion
    of a lasting project property from *any* source. Trust is decided by the
    ledger, not the model.
  - **Routing:** one call per question or deletion description. Its output
    schema is an `enum` of the known slot keys, so the model cannot invent a
    slot. If the user has no slots, it abstains without a call.

  Slot keys are `entity/attribute` (e.g. `payments-service/database`).
  Output is re-validated, malformed claims are skipped, and refusals yield no
  claims and are not cached.

## 12. LLM providers and configuration

`backends.py` defines a `Backend` with one job: send a system and user prompt,
get JSON matching a schema, using each provider's native structured outputs.

| Backend | Provider | How |
|---|---|---|
| `AnthropicBackend` | Claude | `anthropic` SDK, `output_config.format` JSON schema, `effort` (default `low`), server-side refusal fallback. Default model `claude-opus-5-5`. |
| `OpenAIBackend` | OpenAI, and any OpenAI-compatible endpoint (Gemini, Mistral, Groq, Together, Ollama, vLLM, ...) via `OPENAI_BASE_URL` | `openai` SDK, Chat Completions, strict `json_schema` response format; `reasoning_effort` only if set. |

**Configuration** lives in `.env` (git-ignored; template in `.env.example`):

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `anthropic` or `openai`; empty = whichever API key is set (Anthropic first). |
| `LLM_MODEL` | Model id; empty = `claude-opus-5-5` for Anthropic; required for OpenAI. |
| `LLM_EFFORT` | Reasoning effort; empty = `low` for Anthropic, not sent for OpenAI. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Credentials and endpoint. |

Blank values count as unset (the OpenAI SDK would otherwise treat an empty
`OPENAI_BASE_URL` as a real URL).

**Response cache.** Every LLM response is stored in `.cache/llm/`, keyed by a
hash of provider, model, effort, prompt and schema. Rerunning a workload is
free and gives identical results. A cached result reports the token usage of
the original call, so cost comparisons stay honest. Each run's log records the
provider, model, effort, a hash of the prompts, and the counts of calls, cache
hits and refusals.

## 13. Results so far

### Hand-built example: 8 probes, deletion with source ids

| Probe | Type | Expected | no_memory | full_replay | transcript_rag | ledger | ledger_llm |
|---|---|---|---|---|---|---|---|
| p01 | current | PostgreSQL 14 | missed | correct | correct | correct | correct |
| p02 | contamination | main | missed | polluted (contaminated) | polluted (contaminated) | correct | correct |
| p03 | post_update | PostgreSQL 16 | missed | polluted (stale) | polluted (stale) | correct | correct |
| p04 | post_update | make test | missed | polluted (stale) | polluted (stale) | correct | correct |
| p05 | historical | PostgreSQL 14 | missed | polluted (future) | polluted (future) | correct | correct |
| p06 | post_delete | abstain | no_abstention | no_abstention | no_abstention | correct | correct |
| p07 | contamination | main | missed | polluted (contaminated) | polluted (contaminated) | correct | correct |
| p08 | abstention | abstain | no_abstention | no_abstention | no_abstention | correct | correct |
| **Correct** | | | **0/8** | **1/8** | **1/8** | **8/8** | **8/8** |
| Injected tokens | | | 0 | 1,644 | 393 | 103 | 127 |

`ledger_llm` above ran with OpenAI `gpt-4.1-mini`: 14 calls, 4,473 input and
262 output tokens, about 0.84 s mean retrieval latency (versus about 0.1 ms for
the rule-based ledger).

With deletion by description only (`--withhold-sources`), the transcript
baselines get worse on p06 (`leaked`: the deleted hostname comes back, since
they cannot act on a description); the rule-based ledger still scores 8/8.

### Synthetic workload: small preset, seed 0 (99 probes)

| | no_memory | full_replay | transcript_rag | ledger (rules) | ledger_llm |
|---|---|---|---|---|---|
| Correct | 0 | 24 | 34 | 29 | **93** |

By probe type:

- **full_replay** gets every `current` probe right and pollutes every update,
  contamination and historical probe.
- **transcript_rag** is mixed.
- **The rule-based ledger** mostly abstains. Its extractor misses most
  generated phrasings, so it has nothing stored for most facts:
  - 63 false abstentions, including all 18 historical probes.
  - Its correct abstention and post-delete answers come largely from knowing
    nothing, not from correct behaviour.

This is the expected result: the rule-based extractor was a stand-in, and the
generator was built to expose it.

**The LLM ledger** (`gpt-4.1-mini`, 153 extraction calls) scores 93/99, the
same with deletion by description only. It has no polluted, wrong or leaked
answers; all six failures are false abstentions caused by extraction:

- One property split across two slots: the test command went to both
  `ci_test_command` and `pre_push_test_command` (p0033, p0058, p0086).
- A bad attribute name that stuck: a runtime stored as
  `target_java_version = Python 3.12`. The extractor reuses known
  attributes, so the first bad name persists (p0044, p0045).
- An on-call owner never extracted before it was asked about (p0018).

The first run scored 88/99: the slot router returned no slot for questions
worded differently from the slot ("cloud region" vs `region`, "where it
publishes events" vs `message_queue`), mostly historical ones. Telling it to
match on meaning, and never on the service alone, fixed those five without
breaking any abstention. Hiding current values from the router was also tried
and lost recall: a misnamed slot is only recognizable by its value.

## 14. Running it

```
pip install -r requirements.txt
cp .env.example .env                  # then set one provider's key (only ledger_llm needs it)

python -m workloads.examples.build_example                      # rebuild the example timeline
python -m workloads.generator [--preset medium] [--seed N]      # generate a synthetic timeline

python -m runner                                                # free systems, example timeline
python -m runner --withhold-sources                             # deletion by description only
python -m runner --timeline workloads/generated/synthetic-small-s0.json
python -m runner --systems ledger ledger_llm                    # ledger_llm calls the configured LLM
python -m runner --budget 500                                   # tighter evidence budget

python -m pytest -q tests
```

## 15. Tests

67 tests, all offline: the LLM tests use fake clients and never touch the
network or your real `.env`.

| File | Tests | Covers |
|---|---|---|
| `test_schema.py` | 9 | The example is valid and round-trips through JSON. Ground-truth semantics are right. Deleted history stays hidden. The validator rejects a wrong expected answer, a missing forbidden value, an update to the same value, a source that does not match its carrier, and a contradiction from the user. Interface rules hold. |
| `test_runner.py` | 10 | The delivery protocol (session batching, delete timing, no future events) and withholding sources. A perfect "oracle" system scores 8/8. The no-memory lower bound. Full replay's errors are typed correctly. Deletion without sources leaks. The runner truncates over-budget evidence, and RAG fits its own budget. The transcript store is idempotent and deletion sticks. Whole-phrase matching. |
| `test_ledger.py` | 11 | The example scores 8/8 in both deletion modes. Restate, supersede and `as_of`. Disputes from less trusted sources, while a document may still introduce a fact. Deletion erases history and blocks replay, with or without sources. Abstention. User isolation (including a deletion request naming another user's events). Snapshot/restore round-trip. |
| `test_generator.py` | 7 | Same seed gives the same timeline. Ten seeds are valid and clean. Every mutation and probe type appears. Contradictions only target trusted values. Values are unique per user. The hygiene check catches a planted leak. The runner replays generated timelines. |
| `test_llm_extractor.py` | 12 | For both providers: request shape, output filtering (invented event ids, blank and malformed claims), caching, per-provider cache keys, slot routing with an enum, refusals. Provider selection from the environment, and blank values counting as unset. The ledger end to end with an LLM extractor. |

## 16. Known limitations and next steps

### Known limitations

- **Restatement with different spelling.** In the `gpt-4.1-mini` run, the LLM
  extracted "Postgres 14" (e07) as a new value, not a restatement of
  "PostgreSQL 14" (e01). The ledger then recorded an update: an extra item (6
  instead of 5), and a `valid_from` of Jan 20 instead of Jan 5. The probes
  still scored correct (the alias matches), but it is a deduplication (I4)
  miss.
  - **Fix options:** normalize values in the ledger, or let the extractor mark
    a claim as a restatement of a known value.
- **Validity time = event time.** The ledger takes a fact's start time from the
  event that states it. It does not extract a different real-world time from
  the text (e.g. "we migrated last Tuesday").
- **The rule-based extractor** covers four slots and one project per user. It
  is a fallback, not a contender.
- **Deduplication (I4)** is measured only through store size and
  `WriteReport` counts; no probe scores it directly.
- **`multi_fact` probes** are not generated.
- **Scoring is evidence-level.** It does not judge a generated answer.
- **The invariant suite** that checks each system's claimed capabilities
  (referenced in the interface as I1–I7 checks) is not yet a separate
  component; today it is covered by probes plus the tests.

### Next steps

1. Fix slot fragmentation in extraction (split and misnamed slots, see
   section 13), together with the restatement/spelling issue above: both are
   the extractor failing to map a claim onto the slot or value it already
   knows.
2. Make LLM calls deterministic (temperature 0 where the provider allows) and
   run several seeds; every number above comes from one run on one seed.
3. Adapters for third-party memory systems (Mem0, A-MEM, ...) behind the same
   interface.
4. A standalone invariant suite producing the claimed-vs-verified
   capability table (R1).
5. Cost in dollars from a pinned price table; storage-growth curves from
   checkpoint stats; larger (`medium`) workloads and several seeds.

## 17. Glossary

| Term | Meaning |
|---|---|
| **Abstain** | The system says it has no reliable memory for the question. |
| **Alias** | An accepted alternative spelling of a value ("PG 14"). |
| **Carrier event** | The event that expresses a mutation. |
| **Checkpoint** | A point in the stream where probes are asked. |
| **Evidence** | Text the memory system would put into the agent's context. |
| **Extractor** | The part of the ledger that turns text into claims and questions into slots. |
| **Fact / slot** | One property whose value can change, e.g. the database of payments-service. |
| **Forbidden value** | A value of the probed fact that must not appear (stale, future, contaminated or resurrected). |
| **Mutation** | One change to a fact: introduce, update, restate, contradict, delete. |
| **Probe** | A question with a ground-truth expected answer. |
| **Provenance** | Which events a piece of memory came from. |
| **Timeline / workload** | Events + planted facts + mutations + checkpoints + probes. |
| **Tombstone** | A value-free marker that a slot was deleted. |
