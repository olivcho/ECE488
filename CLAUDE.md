# ECE488 Project Conventions

Research question: does an unrolled network trained on one lensless camera transfer to a new camera by swapping in a new PSF?

## Layout
| Path | Holds | In Git |
|---|---|---|
| `src/lensless/` | Reusable, tested code | Yes |
| `scripts/` | Self-contained entry points | Yes |
| `configs/` | YAML run settings | Yes |
| `tests/` | Unit tests | Yes |
| `notebooks/` | Exploration only | Yes |
| `docs/` | Proposal, derivations, references | Yes |
| `experiments/<name>/<YYMMDD>_<technique>_v<N>/` | Config, code version, environment, metrics | Yes, except previews and checkpoints |
| `data/raw/` | Downloaded datasets, never edited | No |
| `data/processed/` | Generated data | No |

## Experiments
- Every result must trace to a run folder, and every run folder records its commit hash and whether the tree had uncommitted changes.
- Do not overwrite a run. A new attempt gets a new version number.
- Store checkpoints outside Git, using Google Drive on Colab.
- Log each experiment in the Obsidian vault at `ECE488 Research/Experiments/EXP-NNN Name.md`, citing the run folder, and add a row to `ECE488 Research Index`. Record failures too.
- Tune on the validation split. Use the test split only for final reported results.

## Code
- Prefer self-contained scripts while exploring. Move code into `src/lensless/` once a method stabilizes.
- Use descriptive names, and comment on why rather than what. Math symbols are fine when they match `docs/admm_updates.md`.
- Keep each function to one task at one level of abstraction.
- Unit-test math, data loading and tensor shapes. Run `uv run pytest` before committing.

## Git
- Commit often with meaningful messages. Keep each pull request under about 400 lines.
- Never add Claude as a co-author or add AI attribution to commits or pull requests.
- Keep large data and checkpoints out of Git.

## Environment
Dependencies are pinned in `uv.lock`. Install with `uv sync`, and add packages with `uv add`.
