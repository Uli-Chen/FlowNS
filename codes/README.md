# FlowNS Research Codebase

This project is now managed with `uv` and logs experiments with `wandb` (TensorBoard removed).

## Environment Setup

```bash
uv sync
```

## Run Training

```bash
uv run python src/main.py
```

Useful `wandb` flags:

```bash
uv run python src/main.py --wandb_mode online --wandb_project FlowNS --wandb_name exp-001
```

- `--wandb_mode offline`: save runs locally in `wandb/` and sync later
- `--wandb_mode online`: upload directly
- `--wandb_mode disabled`: turn off external logging

## Notes

- Entry point: `src/main.py`
- Core training loop: `src/trainer.py`
- Dependencies are centralized in `pyproject.toml`
