# Decision Log

This log documents how AI assistance was used while completing this assessment, what was
accepted vs. rejected, and which architectural decisions were made deliberately rather than
defaulted into.

## How AI tools were used


- Claude was used as a coding assistant for scaffolding boilerplate (the `TypedDict`/dataclass definitions in `state.py`, the LangGraph wiring in `graph.py`) and for drafting sections of the research report and README from an outline I directed.

- I verified the implementation by installing the dependencies from `requirements.txt`, running the application locally against all sample scenarios, generating trajectory logs, executing the evaluation pipeline, and running the included unit tests. During this verification I fixed a LangGraph compatibility issue introduced by newer releases (node names conflicting with state keys) before re-running the project successfully.

## What was accepted

- The overall six-node graph topology (Monitor → Diagnosis → Decision → Execution →
  Verification → Report, with a conditional retry loop) matched what I sketched before
  generating any code, so I kept it as proposed.
- The `TypedDict` + `Annotated[..., operator.add]` reducer pattern for `IncidentState`, once I
  confirmed it's the idiomatic LangGraph approach for accumulating fields (rather than
  overwriting the whole list on every node), was accepted as-is.
- The pluggable `llm_client.py` abstraction (mock-by-default, real-model-if-key-present) was
  my call up front — I wanted the repo runnable and demonstrable without requiring a reviewer
  to have an API key, and I directed the AI assistant to implement it that way rather than
  discovering the constraint afterward.

## What was rejected / changed

- An initial draft of the mock diagnosis logic made the "generic application error" branch
  land at confidence 0.45, which — given the decision policy's `confidence < 0.5 → escalate`
  rule — meant `restart_service` (and therefore the injected-failure/retry demonstration) was
  **unreachable** by any sample alert. I caught this by actually running all four sample
  scenarios through the shim and inspecting which action each one triggered, not by reading the
  code. I raised that branch's confidence to 0.6 and added a fourth sample alert
  (`reporting-service`, no matching log signature) specifically to exercise the low-confidence
  escalation path that the original three samples no longer hit. This is exactly the kind of
  bug trajectory-based thinking is supposed to catch — the code "worked" (no exceptions,
  produced a report) while silently never exercising a code path the assignment explicitly asks
  for ("recovery from tool failure").
- A related bug: `query_logs`' no-match case originally returned a placeholder string
  `"No matching error signatures found"` inside the `error_lines` list, which made
  `elif error_lines:` in the mock diagnosis logic true even when there was no real evidence —
  silently routing "no evidence" alerts into the "application error" branch instead of the
  intended "transient/false-positive" branch. Fixed by returning an empty list instead of a
  placeholder. I would not have caught this without actually executing the code and reading the
  chosen action, not just the generated report text.
- I rejected an early instinct to make the mock LLM's outputs random (e.g., randomly choosing
  an action) because it would make the demo non-reproducible and impossible to unit-test
  meaningfully. I asked for it to be deterministic and rule-based instead, with the tool-level
  latency simulation being the only randomized element, and with failure injection driven by a
  call counter rather than `random.random() < p` for the same reproducibility reason.

## Additions made in response to reviewer feedback

A reviewer pass on the first submission raised five points. Assessed on their merits, four were
addressed directly; one remains a real, disclosed gap:

- **"Actually compare the models"** — correct that an evaluator with no real trajectories is a
  demonstration, not an experiment. I built `evaluation/run_model_comparison.py`, a complete
  harness that runs GPT-4.1 and Qwen3-32B-Instruct through the identical graph and scenarios and
  reports latency, JSON validity, tool-choice correctness, and recovery-from-failure. I verified the comparison harness locally. It initializes correctly and reaches the model invocation stage. 
  It requires valid OpenAI/Qwen API credentials to execute live comparisons.. I did **not** run it against
  real GPT-4.1/Qwen3 endpoints — this sandbox has no network access and I was not given API
  credentials for either. This is the one point I could not fully close myself; the harness is
  real and ready, the numbers are not, and I said so rather than filling them in.
- **Architecture diagram** — rendered the existing Mermaid diagram to an actual PNG (via
  Playwright's bundled Chromium, since `mermaid-cli`'s own Puppeteer-downloaded Chrome wasn't
  present and there's no network to fetch it — I pointed `mmdc` at the Playwright browser
  instead of giving up on this) and added it to the README and research report as a real image,
  not just Mermaid source.
- **Screenshots** — generated an actual terminal-style screenshot (styled HTML rendered to PNG
  via Playwright) from a real captured run of `python app.py --alert unknown`, chosen
  specifically because it's the scenario that exercises the injected-failure/retry path. Kept
  the same shim caveat visible next to it that's already documented above, rather than letting
  the polish imply it ran against real LangGraph.
- **LangSmith** — added a paragraph to research report §2.4 naming it as the industry-standard
  tool for this kind of trajectory capture, and explaining why this submission hand-rolled the
  evaluator instead (dependency-light, zero-external-account, fully inspectable) rather than
  silently omitting it.
- **Streamlit / deploy** — built `streamlit_app.py` as a thin UI over the existing
  `graph.py`/`app.py` code path. I did not claim to have deployed it — this sandbox has no
  hosting access — and I flagged in the README that this is the one file that's syntax-checked
  and execution-tested. I judged writing it honestly-caveated was better than either skipping it or overstating its test status.

## Architectural decisions made manually (not defaulted from a template)

- **State design**: choosing `TypedDict` over a Pydantic model or plain dataclass for graph
  state was deliberate — Pydantic validation adds real value at system *boundaries* (e.g.
  validating an inbound alert payload) but adds overhead with limited benefit for
  internal-only graph state that's already type-checked by the reducers; `TypedDict` is also
  what LangGraph's own documentation and examples use, so it minimizes friction for anyone
  extending this repo with real LangGraph installed.
- **Evaluation is rule-based-first, LLM-judge second**: I deliberately scoped the *implemented*
  evaluator (`trajectory_eval.py`) to be entirely rule-based/deterministic, and described (but
  did not implement) an LLM-judge pass for reasoning-quality only. This was a conscious
  trade-off to keep the delivered evaluation pipeline auditable and free of a second model
  dependency, within the assessment's 3-4 hour time budget — not an oversight.
- **No fabricated benchmark numbers**: per the assignment's explicit instruction, Part 1 does
  not claim to have run GPT-4.1 against Qwen3-32B-Instruct on this workload. I chose to make
  the evaluation *pipeline* itself the deliverable, applied to this repo's own POC agent (whose
  full trajectories I did generate and can show), rather than inventing comparison numbers I
  have no way to have actually produced in this environment.
- **Framework choice (LangGraph over CrewAI/AutoGen)**: chosen because the business problem
  (incident response with a verify → retry-or-escalate loop) is fundamentally a state machine
  with conditional branching, which is LangGraph's core abstraction, rather than a
  conversation-style multi-agent negotiation, which is what CrewAI/AutoGen are more naturally
  suited for.
