import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import toml
import torch

from cell_load.config import ExperimentConfig
from cell_load.data_modules.samplers import PerturbationBatchSampler
from cell_load.utils.data_utils import GlobalH5MetadataCache, H5MetadataCache
from conftest import create_toml_config, make_datamodule


def test_zeroshot_excludes_celltype(synthetic_data):
    """Test that zeroshot cell types are excluded from training."""
    root, cell_types = synthetic_data
    zs_ct = cell_types[0]  # CT0

    config = {
        "datasets": {
            "dataset1": "placeholder"
        },  # Will be updated by create_toml_config
        "training": {"dataset1": "train"},
        "zeroshot": {f"dataset1.{zs_ct}": "test"},
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
        )
        dm.setup()

        # Debug: Check if datasets were created
        print(f"Train datasets: {len(dm.train_datasets)}")
        print(f"Val datasets: {len(dm.val_datasets)}")
        print(f"Test datasets: {len(dm.test_datasets)}")

        # Verify we have some data
        assert len(dm.train_datasets) > 0, "No training datasets created"
        assert len(dm.test_datasets) > 0, "No test datasets created"

        # Verify zeroshot cell type has only control cells in training data
        # (observational data from zeroshot cell types is now included in training)
        train_celltypes = set()
        train_pert_names = set()
        for subset in dm.train_datasets:
            ds = subset.dataset
            for idx in subset.indices:
                ct = ds.get_cell_type(idx)
                train_celltypes.add(ct)
                if ct == zs_ct:
                    # If zeroshot cell type is in training, it should only be control cells
                    pert_name = ds.get_perturbation_name(idx)
                    train_pert_names.add(pert_name)

        print(f"Training cell types: {train_celltypes}")
        print(f"Training perturbations for {zs_ct}: {train_pert_names}")

        # If zeroshot cell type is in training, all should be control cells
        if zs_ct in train_celltypes:
            assert train_pert_names == {"P0"}, (
                f"Zeroshot cell type {zs_ct} should only have control cells in training, "
                f"but found perturbations: {train_pert_names}"
            )

        # Verify zeroshot cell type is in test data
        test_celltypes = set()
        for subset in dm.test_datasets:
            ds = subset.dataset
            for idx in subset.indices:
                test_celltypes.add(ds.get_cell_type(idx))

        print(f"Test cell types: {test_celltypes}")
        assert zs_ct in test_celltypes, (
            f"Zeroshot cell type {zs_ct} not found in test data"
        )

    finally:
        toml_path.unlink()  # Clean up temp file


def test_fewshot_perturbation_splitting(synthetic_data):
    """Test that fewshot perturbations are correctly split between train/val/test."""
    root, cell_types = synthetic_data
    fs_ct = cell_types[1]  # CT1

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
        "fewshot": {
            f"dataset1.{fs_ct}": {
                "val": ["P1", "P2"],
                "test": ["P3", "P4", "P1"],  # P1 appears in both val and test
            }
        },
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
        )
        dm.setup()

        # Debug: Check if datasets were created
        print(f"Train datasets: {len(dm.train_datasets)}")
        print(f"Val datasets: {len(dm.val_datasets)}")
        print(f"Test datasets: {len(dm.test_datasets)}")

        # Verify we have data in all splits
        assert len(dm.train_datasets) > 0, "No training datasets created"
        assert len(dm.val_datasets) > 0, "No validation datasets created"
        assert len(dm.test_datasets) > 0, "No test datasets created"

        # Check that fewshot cell type appears in all splits
        def get_celltypes_in_split(datasets):
            return {
                subset.dataset.get_cell_type(idx)
                for subset in datasets
                for idx in subset.indices
            }

        train_cts = get_celltypes_in_split(dm.train_datasets)
        val_cts = get_celltypes_in_split(dm.val_datasets)
        test_cts = get_celltypes_in_split(dm.test_datasets)

        print(f"Train celltypes: {train_cts}")
        print(f"Val celltypes: {val_cts}")
        print(f"Test celltypes: {test_cts}")

        assert fs_ct in train_cts, f"Fewshot celltype {fs_ct} not in training"
        assert fs_ct in val_cts, f"Fewshot celltype {fs_ct} not in validation"
        assert fs_ct in test_cts, f"Fewshot celltype {fs_ct} not in test"

        # Verify specific perturbations are in correct splits
        def get_perturbations_for_celltype(datasets, target_ct):
            perts = set()
            for subset in datasets:
                ds = subset.dataset
                for idx in subset.indices:
                    if ds.get_cell_type(idx) == target_ct:
                        pert_name = ds.get_perturbation_name(idx)
                        if pert_name != "P0":  # Skip control
                            perts.add(pert_name)
            return perts

        train_perts = get_perturbations_for_celltype(dm.train_datasets, fs_ct)
        val_perts = get_perturbations_for_celltype(dm.val_datasets, fs_ct)
        test_perts = get_perturbations_for_celltype(dm.test_datasets, fs_ct)

        print(f"Train perts for {fs_ct}: {train_perts}")
        print(f"Val perts for {fs_ct}: {val_perts}")
        print(f"Test perts for {fs_ct}: {test_perts}")

        # Val should contain P1, P2
        assert "P1" in val_perts, "P1 not found in validation"
        assert "P2" in val_perts, "P2 not found in validation"

        # Test should contain P1, P3, P4
        assert "P1" in test_perts, "P1 not found in test"
        assert "P3" in test_perts, "P3 not found in test"
        assert "P4" in test_perts, "P4 not found in test"

        # Train should contain remaining perturbations (P5-P9)
        expected_train = {"P5", "P6", "P7", "P8", "P9"}
        assert expected_train.issubset(train_perts), (
            f"Expected train perts {expected_train} not found in {train_perts}"
        )

    finally:
        toml_path.unlink()


def test_training_dataset_behavior(synthetic_data):
    """Test that training datasets include all non-overridden cell types."""
    root, cell_types = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder", "dataset2": "placeholder"},
        "training": {"dataset1": "train", "dataset2": "train"},
        "zeroshot": {"dataset1.CT0": "test"},  # Exclude CT0 from training
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
        )
        dm.setup()

        # Debug info
        print(f"Train datasets: {len(dm.train_datasets)}")
        print(f"Test datasets: {len(dm.test_datasets)}")

        assert len(dm.train_datasets) > 0, "No training datasets created"

        # Get all cell types in training data and check zeroshot behavior
        train_celltypes = set()
        train_ct0_perts = set()
        for subset in dm.train_datasets:
            ds = subset.dataset
            for idx in subset.indices:
                ct = ds.get_cell_type(idx)
                train_celltypes.add(ct)
                if ct == "CT0":
                    # CT0 is zeroshot, should only have control cells in training
                    pert_name = ds.get_perturbation_name(idx)
                    train_ct0_perts.add(pert_name)

        print(f"Training cell types found: {train_celltypes}")
        print(f"CT0 perturbations in training: {train_ct0_perts}")

        # Should include CT1, CT2 from dataset1 and CT3, CT4, CT5 from dataset2
        # CT0 can be in training but only with control perturbations
        expected_celltypes = {"CT0", "CT1", "CT2", "CT3", "CT4", "CT5"}

        # If CT0 is in training (which it should be for observational data),
        # it should only have control cells
        if "CT0" in train_celltypes:
            assert train_ct0_perts == {"P0"}, (
                f"CT0 in training should only have control cells, "
                f"but found perturbations: {train_ct0_perts}"
            )

        assert train_celltypes == expected_celltypes, (
            f"Expected {expected_celltypes}, got {train_celltypes}"
        )

    finally:
        toml_path.unlink()


def test_config_validation():
    """Test that configuration validation works correctly."""
    # Test missing dataset paths
    config = {"training": {"nonexistent_dataset": "train"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config, f)
        toml_path = Path(f.name)

    try:
        experiment_config = ExperimentConfig.from_toml(str(toml_path))
        with pytest.raises(ValueError, match="Missing dataset paths"):
            experiment_config.validate()
    finally:
        toml_path.unlink()

    # Test invalid splits
    config = {
        "datasets": {"dataset1": "/fake/path"},
        "zeroshot": {"dataset1.CT0": "invalid_split"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config, f)
        toml_path = Path(f.name)

    try:
        experiment_config = ExperimentConfig.from_toml(str(toml_path))
        with pytest.raises(ValueError, match="Invalid split"):
            experiment_config.validate()
    finally:
        toml_path.unlink()


def test_sampler_groups(synthetic_data):
    """Test that the sampler correctly groups cells by cell type and perturbation."""
    root, _ = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=3,
            control_pert="P0",
        )
        dm.setup()

        # Check that we have training data
        assert len(dm.train_datasets) > 0, "No training datasets created"

        loader = dm.train_dataloader()
        assert loader is not None, "Train dataloader is None"

        ds = loader.batch_sampler.dataset

        cell_sentence_len = 20
        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=3,
            cell_sentence_len=cell_sentence_len,
            test=False,
            use_batch=False,
        )
        batches = list(sampler)

        for batch in batches:
            # Break batch into sentences of length 20
            for i in range(0, len(batch), cell_sentence_len):
                chunk = batch[i : i + cell_sentence_len]
                ct_codes = set()
                pert_codes = set()

                for gidx in chunk:
                    offset = 0
                    for subset in ds.datasets:
                        if gidx < offset + len(subset):
                            local = subset.indices[gidx - offset]
                            cache = subset.dataset.metadata_cache
                            ct_codes.add(cache.cell_type_codes[local])
                            pert_codes.add(cache.pert_codes[local])
                            break
                        offset += len(subset)

                assert len(ct_codes) == 1, (
                    f"Mixed cell types in chunk {chunk}: {ct_codes}"
                )
                assert len(pert_codes) == 1, (
                    f"Mixed perts in chunk {chunk}: {pert_codes}"
                )

    finally:
        toml_path.unlink()


def test_collate_fn_shapes_and_keys(synthetic_data):
    """Test that the collate function produces correctly shaped outputs."""
    root, _ = synthetic_data

    config = {
        "datasets": {"dataset2": "placeholder"},
        "training": {"dataset2": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path, embed_key="X_hvg", batch_size=4, control_pert="P0"
        )
        dm.setup()

        assert len(dm.train_datasets) > 0, "No training datasets created"

        loader = dm.train_dataloader()
        assert loader is not None, "Train dataloader is None"

        batch = next(iter(loader))

        for key in (
            "pert_cell_emb",
            "ctrl_cell_emb",
            "pert_emb",
            "pert_name",
            "cell_type",
            "cell_type_onehot",
            "batch",
            "batch_name",
        ):
            assert key in batch
        assert isinstance(batch["pert_cell_emb"], torch.Tensor)
        assert batch["pert_cell_emb"].shape[0] == 4
        assert batch["pert_cell_emb"].ndim == 2

        # Test with X_state embedding
        dm = make_datamodule(
            toml_path, embed_key="X_state", batch_size=4, control_pert="P0"
        )
        dm.setup()

        assert len(dm.train_datasets) > 0, (
            "No training datasets created for X_state test"
        )

        loader = dm.train_dataloader()
        assert loader is not None, "Train dataloader is None for X_state test"

        batch = next(iter(loader))

        for key in (
            "pert_cell_emb",
            "ctrl_cell_emb",
            "pert_cell_counts",
            "pert_emb",
            "pert_name",
            "cell_type",
            "cell_type_onehot",
            "batch",
            "batch_name",
        ):
            assert key in batch
        assert isinstance(batch["pert_cell_emb"], torch.Tensor)
        assert batch["pert_cell_emb"].shape[0] == 4
        assert batch["pert_cell_emb"].ndim == 2

    finally:
        toml_path.unlink()


def test_getitem_basal_matches_control(synthetic_data):
    """Test that control cell mappings work correctly."""
    root, _ = synthetic_data

    config = {"datasets": {"dataset1": ""}, "training": {"dataset1": "train"}}

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path, embed_key="X_hvg", batch_size=4, control_pert="P0"
        )
        dm.setup()
        subset = dm.train_datasets[0]
        ds = subset.dataset
        ds.to_subset_dataset(
            "train",
            perturbed_indices=np.array(subset.indices)[
                ds.metadata_cache.pert_codes[subset.indices]
                != ds.metadata_cache.control_pert_code
            ],
            control_indices=np.array(subset.indices)[
                ds.metadata_cache.pert_codes[subset.indices]
                == ds.metadata_cache.control_pert_code
            ],
        )
        sample = ds[subset.indices[0]]
        # Control embedding must be same shape as perturbation embedding and non-negative
        assert sample["ctrl_cell_emb"].shape == sample["pert_cell_emb"].shape
        assert torch.all(sample["ctrl_cell_emb"] >= 0)

    finally:
        toml_path.unlink()


def test_to_subset_dataset_control_flag(synthetic_data):
    """Test the should_yield_control_cells flag."""
    root, _ = synthetic_data

    config = {"datasets": {"dataset1": ""}, "training": {"dataset1": "train"}}

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(toml_path, embed_key="X_hvg", control_pert="P0")
        dm.setup()
        subset = dm.train_datasets[0]
        ds = subset.dataset

        all_idxs = np.array(subset.indices)
        ctrl_mask = (
            ds.metadata_cache.pert_codes[all_idxs]
            == ds.metadata_cache.control_pert_code
        )
        pert_idxs = all_idxs[~ctrl_mask]
        ctrl_idxs = all_idxs[ctrl_mask]

        # By default we yield controls
        full = ds.to_subset_dataset("val", pert_idxs, ctrl_idxs)
        assert len(full.indices) == len(pert_idxs) + len(ctrl_idxs)

        # Turn off yielding controls
        ds.should_yield_control_cells = False
        no_ctrl = ds.to_subset_dataset("val", pert_idxs, ctrl_idxs)
        assert len(no_ctrl.indices) == len(pert_idxs)

    finally:
        toml_path.unlink()


def test_pickle_and_unpickle_dataset(synthetic_data):
    """Test that datasets can be pickled and unpickled."""
    root, _ = synthetic_data

    config = {"datasets": {"dataset1": ""}, "training": {"dataset1": "train"}}

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(toml_path, embed_key="X_hvg", control_pert="P0")
        dm.setup()
        ds = dm.train_datasets[0].dataset

        data = pickle.dumps(ds)
        ds2 = pickle.loads(data)
        # After unpickle, handle must re-open
        assert hasattr(ds2, "h5_file")
        assert ds2.n_cells == ds.n_cells

    finally:
        toml_path.unlink()


def test_invalid_split_name_raises(synthetic_data):
    """Test that invalid split names raise appropriate errors."""
    root, _ = synthetic_data

    config = {"datasets": {"dataset1": ""}, "training": {"dataset1": "train"}}

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(toml_path, embed_key="X_hvg", control_pert="P0")
        dm.setup()
        ds = dm.train_datasets[0].dataset

        with pytest.raises(ValueError):
            ds.to_subset_dataset("invalid_split", np.array([0]), np.array([]))

    finally:
        toml_path.unlink()


def test_H5MetadataCache_parses_categories_and_codes(synthetic_data):
    """Test that H5 metadata caching works correctly."""
    root, cell_types = synthetic_data
    # Pick one file
    fpath = next((root / "dataset1").glob("*.h5"))
    cache = H5MetadataCache(
        str(fpath),
        pert_col="gene",
        cell_type_key="cell_type",
        control_pert="P0",
        batch_col="gem_group",
    )

    # Perturbation categories should be P0..P9
    assert list(cache.pert_categories) == [f"P{i}" for i in range(10)]
    # Only one cell_type category
    assert list(cache.cell_type_categories) == [fpath.stem]
    # Only one batch category
    assert list(cache.batch_categories) == ["batch1"]
    # Codes length matches total cells
    assert cache.n_cells == 10 * 100
    # Control mask sums to 100 (cells with P0)
    assert int(cache.control_mask.sum()) == 100


def test_GlobalH5MetadataCache_singleton_behavior(synthetic_data):
    """Test that global metadata cache behaves as a singleton."""
    root, _ = synthetic_data
    f1 = next((root / "dataset1").glob("*.h5"))
    f2 = next((root / "dataset2").glob("*.h5"))

    gcache = GlobalH5MetadataCache()
    c1 = gcache.get_cache(str(f1), "gene", "cell_type", "P0", "gem_group")
    c1b = gcache.get_cache(str(f1), "gene", "cell_type", "P0", "gem_group")
    c2 = gcache.get_cache(str(f2), "gene", "cell_type", "P0", "gem_group")

    # Same path returns same instance
    assert c1 is c1b
    # Different path returns different instance
    assert c1 is not c2


def test_complex_fewshot_scenario(synthetic_data):
    """Test a complex scenario with multiple fewshot configurations."""
    root, cell_types = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder", "dataset2": "placeholder"},
        "training": {"dataset2": "train"},  # Only dataset2 is for training
        "zeroshot": {
            "dataset1.CT0": "test",  # CT0 entirely goes to test
            "dataset1.CT1": "val",  # CT1 entirely goes to val
        },
        "fewshot": {
            "dataset1.CT2": {  # CT2 is split by perturbations
                "val": ["P1", "P2"],
                "test": ["P8", "P9"],
                # P3-P7 will go to train automatically
            }
        },
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
        )
        dm.setup()

        # Debug info
        print(f"Train datasets: {len(dm.train_datasets)}")
        print(f"Val datasets: {len(dm.val_datasets)}")
        print(f"Test datasets: {len(dm.test_datasets)}")

        # Verify datasets exist
        assert len(dm.train_datasets) > 0, "No training datasets created"
        assert len(dm.val_datasets) > 0, "No validation datasets created"
        assert len(dm.test_datasets) > 0, "No test datasets created"

        # Get cell types in each split
        def get_celltypes_in_split(datasets):
            return {
                subset.dataset.get_cell_type(idx)
                for subset in datasets
                for idx in subset.indices
            }

        train_cts = get_celltypes_in_split(dm.train_datasets)
        val_cts = get_celltypes_in_split(dm.val_datasets)
        test_cts = get_celltypes_in_split(dm.test_datasets)

        print(f"Train celltypes: {train_cts}")
        print(f"Val celltypes: {val_cts}")
        print(f"Test celltypes: {test_cts}")

        # Check perturbations for zeroshot cell types in training
        # CT0 and CT1 can be in training but only as control cells
        train_ct0_perts = set()
        train_ct1_perts = set()
        for subset in dm.train_datasets:
            ds = subset.dataset
            for idx in subset.indices:
                ct = ds.get_cell_type(idx)
                pert = ds.get_perturbation_name(idx)
                if ct == "CT0":
                    train_ct0_perts.add(pert)
                elif ct == "CT1":
                    train_ct1_perts.add(pert)

        # Verify zeroshot cell types only have control perturbations in training
        if "CT0" in train_cts:
            assert train_ct0_perts == {"P0"}, (
                f"CT0 in training should only have control cells, "
                f"but found perturbations: {train_ct0_perts}"
            )
        if "CT1" in train_cts:
            assert train_ct1_perts == {"P0"}, (
                f"CT1 in training should only have control cells, "
                f"but found perturbations: {train_ct1_perts}"
            )
        assert "CT2" in train_cts, "CT2 should be in training (fewshot partial)"
        assert {"CT3", "CT4", "CT5"}.issubset(train_cts), (
            "Training dataset cell types should be in training"
        )

        assert "CT0" not in val_cts, "CT0 should not be in val (goes to test)"
        assert "CT1" in val_cts, "CT1 should be in val (zeroshot to val)"
        assert "CT2" in val_cts, "CT2 should be in val (fewshot partial)"

        assert "CT0" in test_cts, "CT0 should be in test (zeroshot to test)"
        assert "CT1" not in test_cts, "CT1 should not be in test (goes to val)"
        assert "CT2" in test_cts, "CT2 should be in test (fewshot partial)"

        # Verify perturbation splitting for CT2
        def get_perturbations_for_celltype(datasets, target_ct):
            perts = set()
            for subset in datasets:
                ds = subset.dataset
                for idx in subset.indices:
                    if ds.get_cell_type(idx) == target_ct:
                        pert_name = ds.get_perturbation_name(idx)
                        if pert_name != "P0":  # Skip control
                            perts.add(pert_name)
            return perts

        ct2_train_perts = get_perturbations_for_celltype(dm.train_datasets, "CT2")
        ct2_val_perts = get_perturbations_for_celltype(dm.val_datasets, "CT2")
        ct2_test_perts = get_perturbations_for_celltype(dm.test_datasets, "CT2")

        print(f"CT2 train perts: {ct2_train_perts}")
        print(f"CT2 val perts: {ct2_val_perts}")
        print(f"CT2 test perts: {ct2_test_perts}")

        # Check perturbation assignments
        assert {"P1", "P2"}.issubset(ct2_val_perts), (
            f"P1,P2 should be in val, got {ct2_val_perts}"
        )
        assert {"P8", "P9"}.issubset(ct2_test_perts), (
            f"P8,P9 should be in test, got {ct2_test_perts}"
        )
        assert {"P3", "P4", "P5", "P6", "P7"}.issubset(ct2_train_perts), (
            f"P3-P7 should be in train, got {ct2_train_perts}"
        )

    finally:
        toml_path.unlink()


def test_barcode_functionality(synthetic_data):
    """Test that cell barcodes are included when barcode=True."""
    root, cell_types = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        # Test with barcode=True
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
            barcode=True,
        )
        dm.setup()

        # Get a batch from the dataloader
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Check that barcodes are present
        assert "pert_cell_barcode" in batch, "pert_cell_barcode not found in batch"
        assert "ctrl_cell_barcode" in batch, "ctrl_cell_barcode not found in batch"

        # Check that barcodes are lists of strings
        assert isinstance(batch["pert_cell_barcode"], list), (
            "pert_cell_barcode should be a list"
        )
        assert isinstance(batch["ctrl_cell_barcode"], list), (
            "ctrl_cell_barcode should be a list"
        )
        assert len(batch["pert_cell_barcode"]) == len(batch["ctrl_cell_barcode"]), (
            "barcode lists should have same length"
        )

        # Check that barcodes are strings
        for barcode in batch["pert_cell_barcode"]:
            assert isinstance(barcode, str), "barcodes should be strings"
        for barcode in batch["ctrl_cell_barcode"]:
            assert isinstance(barcode, str), "barcodes should be strings"

        # Test with barcode=False (default)
        dm_no_barcode = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=16,
            control_pert="P0",
        )
        dm_no_barcode.setup()

        # Get a batch from the dataloader
        train_loader_no_barcode = dm_no_barcode.train_dataloader()
        batch_no_barcode = next(iter(train_loader_no_barcode))

        # Check that barcodes are not present
        assert "pert_cell_barcode" not in batch_no_barcode, (
            "pert_cell_barcode should not be present when barcode=False"
        )
        assert "ctrl_cell_barcode" not in batch_no_barcode, (
            "ctrl_cell_barcode should not be present when barcode=False"
        )

    finally:
        toml_path.unlink()  # Clean up temp file


def test_group_by_cell_line_batches_contain_single_cell_line(synthetic_data):
    """Test that when group_by_cell_line=True, batches only contain cells from one cell line."""
    root, cell_types = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder", "dataset2": "placeholder"},
        "training": {"dataset1": "train", "dataset2": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=3,
            control_pert="P0",
            group_by_cell_line=True,
        )
        dm.setup()

        loader = dm.train_dataloader()
        assert loader is not None, "Train dataloader is None"

        ds = loader.batch_sampler.dataset
        sampler = loader.batch_sampler

        # Verify that group_by_cell_line is set
        assert sampler.group_by_cell_line == True, "group_by_cell_line should be True"

        batches = list(sampler)

        # Check that each batch contains cells from only one cell line
        for batch_idx, batch in enumerate(batches):
            cell_type_codes = set()

            for gidx in batch:
                offset = 0
                for subset in ds.datasets:
                    if gidx < offset + len(subset):
                        local = subset.indices[gidx - offset]
                        cache = subset.dataset.metadata_cache
                        cell_type_codes.add(cache.cell_type_codes[local])
                        break
                    offset += len(subset)

            assert len(cell_type_codes) == 1, (
                f"Batch {batch_idx} contains cells from multiple cell lines: {cell_type_codes}"
            )

    finally:
        toml_path.unlink()


def test_group_by_cell_line_batches_can_be_shorter_than_batch_size(synthetic_data):
    """Test that batches can be shorter than batch_size when group_by_cell_line=True."""
    root, _ = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        # Use a large batch_size relative to available data
        batch_size = 10
        cell_sentence_len = 20
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=batch_size,
            control_pert="P0",
            cell_sentence_len=cell_sentence_len,
            group_by_cell_line=True,
        )
        dm.setup()

        loader = dm.train_dataloader()
        sampler = loader.batch_sampler

        batches = list(sampler)

        # Check that we have batches shorter than batch_size * cell_sentence_len
        # (not just the last batch)
        expected_max_batch_size = batch_size * cell_sentence_len
        shorter_batches = [
            batch
            for batch in batches
            if len(batch) < expected_max_batch_size
        ]

        # Should have multiple shorter batches (not just the last one)
        assert len(shorter_batches) > 0, (
            "Should have batches shorter than batch_size * cell_sentence_len"
        )

        # Verify that shorter batches are not just at the end
        # (at least one shorter batch should not be the last batch)
        if len(batches) > 1:
            non_last_shorter = [
                i
                for i, batch in enumerate(batches[:-1])
                if len(batch) < expected_max_batch_size
            ]
            assert len(non_last_shorter) > 0, (
                "Should have shorter batches that are not the last batch"
            )

        # Verify that shorter batches still respect single cell line constraint
        ds = loader.batch_sampler.dataset
        for batch in shorter_batches:
            cell_type_codes = set()
            for gidx in batch:
                offset = 0
                for subset in ds.datasets:
                    if gidx < offset + len(subset):
                        local = subset.indices[gidx - offset]
                        cache = subset.dataset.metadata_cache
                        cell_type_codes.add(cache.cell_type_codes[local])
                        break
                    offset += len(subset)
            assert len(cell_type_codes) == 1, (
                f"Shorter batch contains cells from multiple cell lines: {cell_type_codes}. "
                f"Shorter batches should still respect group_by_cell_line constraint."
            )

    finally:
        toml_path.unlink()


def test_group_by_cell_line_mixed_perturbations_in_batch(synthetic_data):
    """Test that batches contain sentences from different perturbations when group_by_cell_line=True."""
    root, _ = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        # Use realistic batch_size to ensure batches contain multiple sentences
        # With batch_size=64 and cell_sentence_len=20, batches can contain up to 64 sentences
        batch_size = 64
        cell_sentence_len = 20
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=batch_size,
            control_pert="P0",
            cell_sentence_len=cell_sentence_len,
            group_by_cell_line=True,
        )
        dm.setup()

        loader = dm.train_dataloader()
        ds = loader.batch_sampler.dataset
        sampler = loader.batch_sampler

        batches = list(sampler)

        # Check that batches contain sentences from multiple perturbations
        batches_with_multiple_perts = 0
        batches_with_multiple_sentences = 0
        for batch_idx, batch in enumerate(batches):
            # Break batch into sentences
            pert_codes_per_sentence = []
            for i in range(0, len(batch), cell_sentence_len):
                chunk = batch[i : i + cell_sentence_len]
                if len(chunk) == cell_sentence_len:  # Only check full sentences
                    pert_codes = set()
                    for gidx in chunk:
                        offset = 0
                        for subset in ds.datasets:
                            if gidx < offset + len(subset):
                                local = subset.indices[gidx - offset]
                                cache = subset.dataset.metadata_cache
                                pert_codes.add(cache.pert_codes[local])
                                break
                            offset += len(subset)
                    pert_codes_per_sentence.append(pert_codes)

            # Count batches with multiple sentences
            if len(pert_codes_per_sentence) > 1:
                batches_with_multiple_sentences += 1
                # Check if this batch has sentences from multiple perturbations
                all_perts = set()
                for pert_set in pert_codes_per_sentence:
                    all_perts.update(pert_set)
                if len(all_perts) > 1:
                    batches_with_multiple_perts += 1

        # With realistic batch sizes, batches should contain multiple sentences
        # Verify that batches with multiple sentences contain sentences from different perturbations
        # This demonstrates that the shuffling mechanism allows mixing across perturbations
        assert batches_with_multiple_sentences > 0, (
            f"Expected batches with multiple sentences, but found none. "
            f"This suggests batch_size may be too small or data distribution is unusual."
        )
        
        # With realistic batch sizes and shuffled sentences, we should see batches with multiple perturbations
        # The shuffling of sentences within each cell type group should result in mixed perturbations
        assert batches_with_multiple_perts > 0, (
            f"Expected batches with multiple sentences to contain multiple perturbations "
            f"(due to shuffling), but found {batches_with_multiple_perts} out of "
            f"{batches_with_multiple_sentences} batches with multiple sentences. "
            f"This suggests sentences may not be properly shuffled across perturbations."
        )
        
        # Verify that a reasonable proportion of multi-sentence batches have multiple perturbations
        # With shuffling, this should be a significant fraction
        ratio = batches_with_multiple_perts / batches_with_multiple_sentences
        assert ratio > 0.3, (
            f"Expected at least 30% of batches with multiple sentences to have multiple perturbations, "
            f"but found {batches_with_multiple_perts}/{batches_with_multiple_sentences} ({ratio:.1%}). "
            f"This suggests sentences may not be properly shuffled across perturbations."
        )

    finally:
        toml_path.unlink()


def test_group_by_cell_line_sentences_shuffled_within_batch(synthetic_data):
    """Test that sentences are shuffled within cell type groups before batching."""
    root, _ = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        # Use realistic batch_size to ensure batches contain multiple sentences
        batch_size = 64
        cell_sentence_len = 20
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=batch_size,
            control_pert="P0",
            cell_sentence_len=cell_sentence_len,
            group_by_cell_line=True,
        )
        dm.setup()

        loader = dm.train_dataloader()
        ds = loader.batch_sampler.dataset
        sampler = loader.batch_sampler

        # Check shuffling by examining the sentences directly (before they're flattened into batches)
        # Sentences are grouped by cell type and shuffled within each group
        # We can verify shuffling by checking that sentences from the same cell type
        # are not in sorted order by perturbation
        
        # Group sentences by cell type
        sentences_by_cell_type = {}
        for sentence in sampler.sentences:
            cell_type_code = sampler._get_cell_type_code_for_sentence(sentence)
            if cell_type_code not in sentences_by_cell_type:
                sentences_by_cell_type[cell_type_code] = []
            sentences_by_cell_type[cell_type_code].append(sentence)
        
        # For each cell type, check that sentences are shuffled
        # by verifying they're not in sorted order by perturbation
        cell_types_with_shuffled_sentences = 0
        for cell_type_code, sentences in sentences_by_cell_type.items():
            if len(sentences) < 2:
                continue
            
            # Get perturbation codes for each sentence
            pert_codes_per_sentence = []
            for sentence in sentences:
                # Get perturbation code from first cell in sentence
                first_cell_idx = sentence[0]
                offset = 0
                for subset in ds.datasets:
                    if first_cell_idx < offset + len(subset):
                        local = subset.indices[first_cell_idx - offset]
                        cache = subset.dataset.metadata_cache
                        pert_codes_per_sentence.append(cache.pert_codes[local])
                        break
                    offset += len(subset)
            
            # Check if sorted (would indicate no shuffling)
            if len(pert_codes_per_sentence) > 1:
                is_sorted = all(
                    pert_codes_per_sentence[i] <= pert_codes_per_sentence[i + 1]
                    for i in range(len(pert_codes_per_sentence) - 1)
                )
                if not is_sorted:
                    cell_types_with_shuffled_sentences += 1
        
        # Verify that sentences are shuffled within cell type groups
        # With multiple cell types and shuffled sentences, we should see non-sorted order
        assert len(sentences_by_cell_type) > 0, "Expected sentences grouped by cell type"
        
        cell_types_with_multiple_sentences = sum(
            1 for sentences in sentences_by_cell_type.values() 
            if len(sentences) >= 2
        )
        
        assert cell_types_with_multiple_sentences > 0, (
            f"Expected cell types with multiple sentences, but found none. "
            f"This suggests data distribution is unusual."
        )
        
        # With shuffling, at least some cell types should have non-sorted sentence order
        assert cell_types_with_shuffled_sentences > 0, (
            f"Expected shuffled sentences within cell type groups, but found "
            f"{cell_types_with_shuffled_sentences} cell types with shuffled order out of "
            f"{cell_types_with_multiple_sentences} cell types with multiple sentences. "
            f"This suggests sentences may not be shuffled."
        )
        
        # Verify that a reasonable proportion of cell types show shuffling
        ratio = cell_types_with_shuffled_sentences / cell_types_with_multiple_sentences
        assert ratio > 0.5, (
            f"Expected at least 50% of cell types with multiple sentences to show shuffled order, "
            f"but found {cell_types_with_shuffled_sentences}/{cell_types_with_multiple_sentences} ({ratio:.1%}). "
            f"This suggests sentences may not be properly shuffled."
        )

    finally:
        toml_path.unlink()


def test_group_by_cell_line_all_cells_included_in_batches(synthetic_data):
    """Test that all cells from the dataset are included in at least one batch when group_by_cell_line=True.
    
    Note: This test only applies when drop_last=False (the default). When drop_last=True,
    the last incomplete batch may be dropped, so some cells may not appear in any batch.
    """
    root, _ = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder", "dataset2": "placeholder"},
        "training": {"dataset1": "train", "dataset2": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        # Use drop_last=False (default) to ensure all cells are included
        batch_size = 64
        cell_sentence_len = 20
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=batch_size,
            control_pert="P0",
            cell_sentence_len=cell_sentence_len,
            group_by_cell_line=True,
            drop_last=False,  # Explicitly set to False to test complete coverage
        )
        dm.setup()

        loader = dm.train_dataloader()
        ds = loader.batch_sampler.dataset
        sampler = loader.batch_sampler

        # Verify drop_last is False
        assert sampler.drop_last == False, "Test requires drop_last=False"

        # Get all cells that should be in batches (all global indices in the dataset)
        all_expected_cells = set()
        global_offset = 0
        for subset in ds.datasets:
            for i in range(len(subset)):
                all_expected_cells.add(global_offset + i)
            global_offset += len(subset)

        # Collect all cells that appear in batches
        all_cells_in_batches = set()
        batches = list(sampler)
        for batch in batches:
            all_cells_in_batches.update(batch)

        # Verify that all expected cells appear in at least one batch
        missing_cells = all_expected_cells - all_cells_in_batches
        assert len(missing_cells) == 0, (
            f"Expected all cells to appear in batches, but {len(missing_cells)} cells are missing. "
            f"Total expected: {len(all_expected_cells)}, found in batches: {len(all_cells_in_batches)}. "
            f"This suggests cells are being dropped when they shouldn't be (drop_last=False)."
        )

        # Also verify we don't have any extra cells (cells not in the dataset)
        extra_cells = all_cells_in_batches - all_expected_cells
        assert len(extra_cells) == 0, (
            f"Found {len(extra_cells)} cells in batches that are not in the dataset. "
            f"This suggests an indexing error."
        )

        # Verify counts match
        assert len(all_cells_in_batches) == len(all_expected_cells), (
            f"Cell count mismatch: expected {len(all_expected_cells)} cells, "
            f"found {len(all_cells_in_batches)} cells in batches."
        )

    finally:
        toml_path.unlink()


def test_group_by_cell_line_backwards_compatibility(synthetic_data):
    """Test that default behavior (group_by_cell_line=False) is unchanged."""
    root, _ = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder", "dataset2": "placeholder"},
        "training": {"dataset1": "train", "dataset2": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        # Test with group_by_cell_line=False (default)
        dm_false = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=3,
            control_pert="P0",
            group_by_cell_line=False,
        )
        dm_false.setup()

        # Test without specifying group_by_cell_line (should default to False)
        dm_default = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=3,
            control_pert="P0",
        )
        dm_default.setup()

        loader_false = dm_false.train_dataloader()
        loader_default = dm_default.train_dataloader()

        assert loader_false.batch_sampler.group_by_cell_line == False
        assert loader_default.batch_sampler.group_by_cell_line == False

        # Both should generate batches (may have different ordering due to shuffling)
        batches_false = list(loader_false.batch_sampler)
        batches_default = list(loader_default.batch_sampler)

        assert len(batches_false) > 0, "Should generate batches"
        assert len(batches_default) > 0, "Should generate batches"

        # The key test is that group_by_cell_line=False doesn't enforce
        # single cell type per batch (unlike when True)
        # We verify this by comparing behavior: when True, batches are restricted
        # to single cell type; when False, there's no such restriction
        
        # Compare with group_by_cell_line=True to show the difference
        dm_true = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=3,
            control_pert="P0",
            group_by_cell_line=True,
        )
        dm_true.setup()
        loader_true = dm_true.train_dataloader()
        batches_true = list(loader_true.batch_sampler)
        
        # Both should generate batches
        assert len(batches_true) > 0, "Should generate batches with group_by_cell_line=True"
        
        # The key difference: with group_by_cell_line=True, each batch is restricted
        # to a single cell type. With False, there's no such restriction.
        # We've already verified this in test_group_by_cell_line_batches_contain_single_cell_line
        # Here we just verify the parameter is correctly set and defaults work

    finally:
        toml_path.unlink()


def test_group_by_cell_line_sampler_helper_methods(synthetic_data):
    """Test the helper methods for getting cell type codes."""
    root, _ = synthetic_data

    config = {
        "datasets": {"dataset1": "placeholder"},
        "training": {"dataset1": "train"},
    }

    toml_path = create_toml_config(root, config)

    try:
        dm = make_datamodule(
            toml_path,
            embed_key="X_hvg",
            batch_size=3,
            control_pert="P0",
            group_by_cell_line=True,
        )
        dm.setup()

        loader = dm.train_dataloader()
        ds = loader.batch_sampler.dataset
        sampler = loader.batch_sampler

        # Test _get_cell_type_code_for_global_idx
        # Get a global index from the dataset
        if len(ds.datasets) > 0 and len(ds.datasets[0]) > 0:
            global_idx = 0
            cell_type_code = sampler._get_cell_type_code_for_global_idx(global_idx)
            assert isinstance(cell_type_code, (int, np.integer)), (
                "Cell type code should be an integer"
            )

            # Test _get_cell_type_code_for_sentence
            # Get a sentence from the sampler
            if len(sampler.sentences) > 0:
                sentence = sampler.sentences[0]
                sentence_cell_type_code = sampler._get_cell_type_code_for_sentence(sentence)
                assert isinstance(sentence_cell_type_code, (int, np.integer)), (
                    "Sentence cell type code should be an integer"
                )

                # Verify that all cells in the sentence have the same cell type
                for gidx in sentence:
                    ct_code = sampler._get_cell_type_code_for_global_idx(gidx)
                    assert ct_code == sentence_cell_type_code, (
                        f"All cells in sentence should have same cell type code. "
                        f"Expected {sentence_cell_type_code}, got {ct_code} for index {gidx}"
                    )

        # Test error handling
        with pytest.raises(ValueError, match="out of range"):
            sampler._get_cell_type_code_for_global_idx(999999)

        with pytest.raises(ValueError, match="Empty sentence"):
            sampler._get_cell_type_code_for_sentence([])

    finally:
        toml_path.unlink()