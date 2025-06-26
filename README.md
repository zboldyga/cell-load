# Cell Load

A PyTorch-based data loading library for single-cell perturbation data.

## Features

- Load perturbation data from H5 files (AnnData format)
- Support for multiple cell types per dataset
- Configurable mapping strategies for control cell selection
- Zero-shot and few-shot learning support
- Cell barcode tracking (optional)

## Installation

```bash
pip install cell-load
```

## Quick Start

### 1. Create a TOML configuration file:

The TOML configuration file defines your datasets, training splits, and experimental setup. Here's the format:

```toml
# Dataset paths - maps dataset names to their directories
[datasets]
replogle = "/path/to/replogle_dataset/"
jurkat = "/path/to/jurkat_dataset/"

# Training specifications
# All cell types in a dataset automatically go into training (excluding zeroshot/fewshot overrides)
[training]
replogle = "train"
jurkat = "train"

# Zeroshot specifications - entire cell types go to val or test
[zeroshot]
"replogle.jurkat" = "test"
"jurkat.rpe1" = "val"

# Fewshot specifications - explicit perturbation lists
[fewshot]

[fewshot."replogle.rpe1"]
val = ["AARS"]
test = ["AARS", "NUP107", "RPUSD4"]  # can overlap with val
# train gets all other perturbations automatically

[fewshot."jurkat.k562"]
val = ["GENE1", "GENE2"]
test = ["GENE3", "GENE4"]
```

#### TOML Configuration Format

**`[datasets]`**: Maps dataset names to their directory paths
- Each dataset should contain H5 files (one per cell type)
- Files should be named like `cell_type.h5` or `cell_type.h5ad`

**`[training]`**: Specifies which datasets are used for training
- Set to `"train"` to include all cell types in training (except those in zeroshot/fewshot)

**`[zeroshot]`**: Holds out entire cell types for testing
- Format: `"dataset.cell_type" = "split"`
- Split can be `"val"` or `"test"`
- Example: `"replogle.jurkat" = "test"` holds out all Jurkat cells from training

**`[fewshot]`**: Holds out specific perturbations within cell types
- Format: `[fewshot."dataset.cell_type"]`
- `val = ["pert1", "pert2"]`: Perturbations for validation
- `test = ["pert3", "pert4"]`: Perturbations for testing
- Remaining perturbations go to training

### 2. Command Line Usage

The most common parameters for data loading are:

```bash
# Basic required parameters
data.kwargs.toml_config_path=/path/to/config.toml
data.kwargs.embed_key=X_hvg
data.kwargs.num_workers=24
data.kwargs.batch_col=gem_group
data.kwargs.pert_col=gene
data.kwargs.cell_type_key=cell_type
data.kwargs.control_pert=non-targeting

# Optional parameters
data.kwargs.barcode=true  # Enable cell barcode output
data.kwargs.perturbation_features_file=/path/to/gene_embeddings.pt
data.kwargs.output_space=gene
data.kwargs.basal_mapping_strategy=random
data.kwargs.n_basal_samples=1
data.kwargs.should_yield_control_cells=true
```

### 3. Programmatic Usage

```python
from cell_load.data_modules import PerturbationDataModule

dm = PerturbationDataModule(
    # Required parameters
    toml_config_path="/path/to/config.toml",
    embed_key="X_hvg",
    num_workers=24,
    batch_col="gem_group",
    pert_col="gene",
    cell_type_key="cell_type",
    control_pert="non-targeting",
    
    # Optional parameters
    barcode=True,  # Enable cell barcode output
    perturbation_features_file="/path/to/gene_embeddings.pt",
    output_space="gene",
    basal_mapping_strategy="random",
    n_basal_samples=1,
    should_yield_control_cells=True,
    batch_size=128,
)
dm.setup()

# Get training data
train_loader = dm.train_dataloader()
for batch in train_loader:
    # batch contains:
    # - pert_cell_emb: perturbed cell embeddings
    # - ctrl_cell_emb: control cell embeddings
    # - pert_emb: perturbation one-hot encodings or embeddings
    # - pert_name: perturbation names
    # - cell_type: cell types
    # - batch: batch information
    # - pert_cell_barcode: cell barcodes (if barcode=True)
    # - ctrl_cell_barcode: control cell barcodes (if barcode=True)
    pass
```

## Parameter Reference

### Required Parameters

- **`toml_config_path`**: Path to the TOML configuration file defining datasets and splits
- **`embed_key`**: Key in the H5 file's `obsm` section to use for cell embeddings (e.g., "X_hvg", "X_state")
- **`pert_col`**: Column name in `obs` for perturbation information (default: "gene")
- **`cell_type_key`**: Column name in `obs` for cell type information (default: "cell_type")
- **`batch_col`**: Column name in `obs` for batch/plate information (default: "gem_group")
- **`control_pert`**: Value in `pert_col` that represents control cells (default: "non-targeting")

### Optional Parameters

- **`barcode`**: If `true`, include cell barcodes in output (default: `false`)
- **`perturbation_features_file`**: Path to .pt file containing pre-computed gene embeddings
- **`output_space`**: Output space for model predictions ("gene" or "all", default: "gene")
- **`basal_mapping_strategy`**: Strategy for mapping perturbed cells to controls ("batch" or "random", default: "random")
- **`n_basal_samples`**: Number of control cells to sample per perturbed cell (default: 1)
- **`should_yield_control_cells`**: Include control cells in output (default: `true`)
- **`num_workers`**: Number of workers for data loading (default: 8)
- **`batch_size`**: Batch size for training (default: 128)

### Usage

To enable cell barcode output, add the following to your command line:

```bash
data.kwargs.barcode=true
```

Or when creating the data module programmatically:

```python
from cell_load.data_modules import PerturbationDataModule

dm = PerturbationDataModule(
    toml_config_path="config.toml",
    # ... other parameters
)
```

## Advanced Configuration

### Zero-shot Learning

To set up zero-shot learning (entire cell types held out for testing):

```toml
[zeroshot]
"dataset.cell_type" = "test"
```

### Few-shot Learning

To set up few-shot learning (specific perturbations held out):

```toml
[fewshot."dataset.cell_type"]
val = ["pert1", "pert2"]
test = ["pert3", "pert4"]
```

### Custom Perturbation Features

To use pre-computed gene embeddings instead of one-hot encodings:

```bash
data.kwargs.perturbation_features_file=/path/to/gene_embeddings.pt
```

The .pt file should contain a dictionary mapping gene names to embedding vectors.
