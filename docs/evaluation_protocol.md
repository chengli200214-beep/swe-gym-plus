# Evaluation protocol

1. Freeze the dataset revision and write one normalized manifest.
2. Group related instances by repository/PR and assign `smoke`, `dev`, `eval`
   and `train` before sampling trajectories.
3. Run model-free controls: base + test patch should fail, and base + gold
   patch + test patch should pass. A failed build or missing test command is
   `blocked`, never a simulated pass.
4. Run the same model/Harness configuration for `k=1`, `k=2` and `k=4` when
   comparing inference budget. Do not reuse a writable workspace between
   candidates.
5. Select from visible evidence, then run the selected patch in a fresh formal
   evaluation workspace.
6. Report formal pass rate, actual selection success, oracle coverage@k,
   tokens, cost, duration, runtime failures, and recovery decisions.

The current test suite proves contracts and a deterministic fixture only. It
does not claim a SWE-Gym solving rate.
