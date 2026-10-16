# Coding Agent Instructions

## Primary objective

Implement requested changes correctly and efficiently. Minimize unnecessary repository exploration, execution, testing, and documentation maintenance.

These instructions take priority over optional workflows. Do not interpret a request to implement a change as authorization to run a full experiment.

## Environment

- Project root: `/scratch/jhu/uchitra1/users/ajain71/isodepth-statistical-testing`
- Same tree via symlink: `/home/jhu/ajain71/scratch_uchitra1/users/ajain71/isodepth-statistical-testing`
- This file's directory is that project. Do not search `/home/jhu/ajain71`, `miniforge3`, `.cursor`, or agent transcripts to locate it.
- Cursor loads `AGENTS.md`, not this filename. Keep the two copies the same.
- Miniforge: `/home/jhu/ajain71/miniforge3`.
- Python environment: `isodepth_env`.
- R environment: `r_env`.
- Reuse existing environments; never create or install dependencies without permission.
- Once the working environment is identified, do not repeat environment discovery unnecessarily.
- Main experiment entry point: `python run_permutation.py --config <config-path>`.

## GPU / SLURM

- The project uses PyTorch and may require GPU execution.
- Use `l40s` for project GPU jobs rather than the more expensive `a100`.
- Production SLURM jobs require the appropriate account and billing flags, including `--account=uchitra1` and `--comment=accept_cost`.
- Do not automatically submit SLURM jobs.
- Check CUDA availability within the actual execution environment before substantial GPU work.
- If GPU execution is intended but unavailable, report the issue rather than silently falling back to CPU for an expensive workload.
- Tiny CPU tests are acceptable for basic correctness checks.

## Execution and testing

- Make focused changes with minimal unrelated modifications.
- For straightforward edits, avoid exhaustive repository exploration.
- Always inspect modified code for obvious mistakes.
- Default to static validation and focused, inexpensive tests.
- Use small synthetic datasets for algorithmic smoke tests.
- Default test budget: approximately 30 seconds of execution.
- Do not run full test suites, extensive benchmarks, real-data experiments, model-training sweeps, or hundreds of permutations without explicit authorization.
- Avoid rerunning tests that already passed unless relevant code changed.
- Do not repeatedly retry environmental failures.
- Do not create extensive test infrastructure for trivial changes.

## Scientific correctness

- Preserve statistical assumptions and mathematical behavior unless changes are explicitly requested.
- Do not silently modify experiment parameters or statistical estimators for convenience.
- Verify units carefully, particularly native coordinates versus micrometers.
- Do not interpret a successful smoke test as evidence of statistical calibration or scientific validity.

## Documentation and memory

- `AGENTS.md` is an operational guide, not a history of agent activity.
- Do not automatically append findings, debugging logs, experiment results, or implementation summaries to this file.
- Consult relevant documentation under `docs/` only when the task requires it.
- Read the smallest relevant documentation section rather than all historical notes.
- Treat current code and configuration files as authoritative for implemented behavior.
- Update documentation only when a change makes existing documentation inaccurate or introduces an important durable convention.
- Prefer editing or replacing an existing entry over appending another.
- Never record transient errors, numerical experiment outcomes, or one-off file paths in `AGENTS.md`.
- Do not edit documentation unrelated to the task.

## Completion

Stop when the requested change is implemented and proportionately validated.

Report briefly:
- What changed.
- What was tested.
- Any remaining limitations.

Do not undertake additional improvements unless requested.