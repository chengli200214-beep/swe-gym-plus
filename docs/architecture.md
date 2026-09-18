# CodeAgentBench architecture

```text
SWE-Gym row
  -> task manifest + grouped split
  -> AgentTaskView ---------------------> model / bash runtime
       (no gold patch)                         |
                                               v
                                  workspace + intent/receipt journal
                                               |
                                  checkpoint + diff + trajectory
                                               v
                      candidates -- evidence-only selector -- selected diff
                                               |
                              fresh evaluation workspace + EvalSpec
                                               v
                                   formal verdict / cost / report
```

## Reliable execution

The runtime writes a checkpoint before a tool action with the workspace digest
and pending action id. `BashExecutor` writes the intent before starting the
process and the receipt after it finishes. On restart, `ActionJournal` compares
the current digest with the pending intent. A read-only action can be retried;
an unacknowledged side-effecting action is stopped because its outcome is
unknown. This is intentionally narrower than claiming arbitrary container
processes are resumable.

## Independent evaluation

The agent never receives `gold_patch`, evaluation logs or later git history.
The evaluator creates a fresh workspace from `repo/base_commit`, applies the
candidate patch and test patch, then runs the formal test command. A zero exit
code is necessary but is not treated as evidence that a hidden answer was
matched. Structured `fail_to_pass`, `pass_to_pass`, failed tests, stdout and
stderr are retained for review.

## Multi-rollout accounting

Candidates are generated independently from the same task/base state. The
selector ranks evidence before formal labels exist. Reports distinguish:

- actual selection success: selected candidate passed formal evaluation;
- oracle coverage@k: at least one generated candidate passed;
- total cost: all candidate calls, retries, selector work and test time.

The demo uses a local runner. Docker/PostgreSQL are deployment options described
in `compose.yaml`; the default code path remains runnable on a CPU laptop.
