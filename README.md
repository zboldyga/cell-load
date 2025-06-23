# cell-load

> Dataloaders for state-sets models

## Installation

Distributed via [`uv`](https://docs.astral.sh/uv).

```bash
# Clone repo
git clone github.com:arcinstitute/cell-load
cd cell-load

# Initialize venv
uv venv

# Install
uv pip install -e .
```

## Data Module Usage

This is currently set up to do context generalization - users can specify multiple AnnData files,
and can identify specific cell lines to add to the test set. For those test cell lines, users can
identify specific perturbations to hold out for testing.

For example usage please see the [`State`](https://github.com/ArcInstitute/state) repository.
