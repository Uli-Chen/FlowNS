import os


def load_m0_checkpoint_path(results_dir):
    """Return the absolute M0 checkpoint path recorded by pilot_m0_baseline."""
    pointer_path = os.path.join(results_dir, 'M0_model_path.txt')
    if not os.path.exists(pointer_path):
        raise FileNotFoundError(
            f'M0 checkpoint pointer not found: {pointer_path}. Run M0 first.'
        )

    with open(pointer_path) as f:
        checkpoint_path = f.read().strip()

    if not checkpoint_path:
        raise ValueError(f'M0 checkpoint pointer is empty: {pointer_path}')

    if not os.path.isabs(checkpoint_path):
        project_dir = os.path.abspath(os.path.join(results_dir, '..', '..'))
        checkpoint_path = os.path.join(project_dir, checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'M0 checkpoint not found: {checkpoint_path}. Re-run M0.'
        )
    return checkpoint_path
