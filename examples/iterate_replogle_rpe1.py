import os
import sys
from typing import Any


def _ensure_src_on_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.abspath(os.path.join(repo_root, "..", "src"))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()

from cell_load.data_modules import PerturbationDataModule  # noqa: E402


def _shape_of(x: Any):
    try:
        return tuple(x.shape)
    except Exception:
        return type(x).__name__


def main() -> None:
    print("Starting loading process")
    toml_config = \
        "/large_storage/ctc/ML/state_sets/replogle_proper/rpe1_zeroshot.toml"

    dm = PerturbationDataModule(
        toml_config_path=toml_config,
        batch_col="ct_gem_group",
        control_pert="non-targeting",
        basal_mapping_strategy="batch",
        # Optional runtime knobs; tweak as needed
        num_workers=4,
        batch_size=8,
    )

    import time
    start_time = time.time()
    dm.setup()
    elapsed = time.time() - start_time
    print(f"Done setting up datamodule (took {elapsed:.2f} seconds)")

    start_loader_time = time.time()
    train_loader = dm.train_dataloader()
    loader_elapsed = time.time() - start_loader_time
    print(f"Time to get train_loader: {loader_elapsed:.2f} seconds")
    print(f"Control perturbation: {dm.get_control_pert()}")

    num_batches = 25
    batch_start = time.time()
    for i, batch in enumerate(train_loader):
        shapes = {k: _shape_of(v) for k, v in batch.items()}
        if i + 1 >= num_batches:
            break
    batch_elapsed = time.time() - batch_start
    print(f"Time to process batches: {batch_elapsed:.3f} seconds")


if __name__ == "__main__":
    main()


