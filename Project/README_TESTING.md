# Testing and Evaluation

Everything here runs with **no API key and no network**.

Prerequisites, run once: entry points 2 to 5.

```bash
python3 run_database_setup.py && python3 run_data_preparation.py \
  && python3 run_model_training.py && python3 run_knowledge_indexing.py --backend offline
```

> The test suites pin the **offline** embedder. If you built the index with
> `--backend minilm`, rebuild it with `--backend offline` before testing. The index
> refuses to be queried by an embedder other than the one that built it.

## 1. Specification suite: `tests.py` (50 tests)

```bash
python3 -m pytest tests.py -q        # → 50 passed
```

This is the supplied suite, and it is **not modified**. It checks cleaning, models,
the knowledge base, business rules, routing, memory, security, persistence, the
workflow's structure and the REST service. Nothing in it asks a language model
anything.

## 2. Regression suite: `tests_regression.py` (62 tests)

```bash
python3 -m pytest tests_regression.py -q   # → 62 passed
```

These are extra tests written against the defects the specification warns about.
Everything that writes runs against a **sandbox database and checkpoint file in a
temporary folder**, with the offline model, so your queue and audit log stay
clean. Run it **separately** from `tests.py`.

Highlights:

| Area | What is guarded |
|---|---|
| Cleaning | a sentinel is blanked rather than clamped; the median is used, not the mean; categories are never filled with the commonest value; the serving feed never carries the answer; the training cleaner cleans `"1,231.97 INR"` |
| Models | single-row order matches the schema; a missing file names the step to run; the metrics file lists all 13 candidates |
| Rules | a value just under a threshold lands in the lower band; an unknown tier takes the Standard row; SLA compensation is capped; a negative order value is refused |
| Knowledge | headings detected; no word split across chunks; overlap; stable labels; cosine distance; an embedder mismatch is refused |
| LLM layer | labels parsed whatever the decoration; sections stop at the next label; the offline client uses the prompt's figures |
| Routing | `hi` never matches inside `this`; a greeting on a complaint is still a complaint; "arrived cracked" is damage, not a status question |
| Chat | a complaint followed by its order number files **one** ticket, unassigned |
| Security | tampered and expired tokens are refused; an unknown role holds nothing |
| Workflow | inside authority it takes 6 calls with no pause; a paused run **survives a restart**, keeps its risk figures and recomputes its estimate; a reject takes 5 calls; an unknown order spends 0 calls |
| API | a manager can't approve above their own limit (403 names admin); an admin can; a second review gets 409; customers don't see desk notes |
| Ingestion | a failed pass never moves the watermark; the evaluation file is refused; upsert by order |
| Runtime | every file passes the Python 3.9 check |

## 3. Evaluation harness: `run_evaluation.py`

```bash
python3 run_evaluation.py                 # offline (default, deterministic)
python3 run_evaluation.py --backend auto  # use Gemini if a key is set (spends calls)
```

It replays the golden sets in `evaluation/golden_sets/` against a sandbox
database, writes `output/evaluation_report.json`, and **exits non-zero if any gated
bar is missed**, so it can run in CI.

```
CHECK                                            VALUE     BAR  RESULT
Retrieval finds the right document              0.9600    0.85  PASS
Every citation is real                          1.0000    1.00  PASS
Declines what it cannot answer                  1.0000    0.90  PASS
Router covers known phrasing                    1.0000    0.85  PASS
Router generalises to new phrasing              0.2670    0.25  PASS   (reported, not gated)
A later turn remembers an earlier one           1.0000    0.90  PASS
Never offers what policy forbids                0.0000    0.00  PASS
Every draft cites its policy basis              0.0000    0.00  PASS
No run breaches its call budget                 0.0000    0.00  PASS
```

**Read the two routing lines together.** A keyword router scores high on the
wording it was built for and much lower on wording it has never seen. That gap is
the honest measure of the rule layer; closing it is the job of the language-model
routing layer, which the offline backend can't stand in for. So the held-out
figure is reported but doesn't fail the run.

## 4. Python 3.9 guard

```bash
python3 scripts/check_python39_syntax.py
# Checking 72 source files against Python 3.9
# All clear. Every file parses and every name exists on 3.9.
```

It runs three checks: it parses each file with the 3.9 grammar, looks for `X | Y`
union annotations, and searches for standard-library names added after 3.9
(`datetime.UTC`, `typing.Self`, and so on).
