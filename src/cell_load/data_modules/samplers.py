import logging
import time
from typing import Iterator
import copy

import numpy as np
from torch.utils.data import Sampler, Subset
import torch.distributed as dist

from ..dataset import MetadataConcatDataset, PerturbationDataset
from ..utils.data_utils import H5MetadataCache

logger = logging.getLogger(__name__)


class PerturbationBatchSampler(Sampler):
    """
    Samples batches ensuring that cells in each batch share the same
    (cell_type, perturbation) combination, using only H5 codes.

    Instead of grouping by cell type and perturbation names, this sampler
    groups based on integer codes stored in the H5 file (e.g. `cell_type_codes`
    and `pert_codes` in the H5MetadataCache). This avoids repeated string operations.

    Supports distributed training.
    """

    def __init__(
        self,
        dataset: "MetadataConcatDataset",
        batch_size: int,
        drop_last: bool = False,
        cell_sentence_len: int = 512,
        test: bool = False,
        use_batch: bool = False,
        seed: int = 0,
        epoch: int = 0,
        group_by_cell_line: bool = False,
        shuffle_batches_per_epoch: bool = False,
    ):
        logger.info(
            "Creating perturbation batch sampler with metadata caching (using codes)..."
        )
        start_time = time.time()

        # If the provided dataset has a `.data_source` attribute, use that.
        self.dataset = (
            dataset.data_source if hasattr(dataset, "data_source") else dataset
        )
        self.batch_size = batch_size
        self.test = test
        self.use_batch = use_batch
        self.seed = seed
        self.epoch = epoch

        if self.test and self.batch_size != 1:
            logger.warning(
                "Batch size should be 1 for test mode. Setting batch size to 1."
            )
            self.batch_size = 1

        self.cell_sentence_len = cell_sentence_len
        self.drop_last = drop_last
        self.group_by_cell_line = group_by_cell_line
        self.shuffle_batches_per_epoch = shuffle_batches_per_epoch

        logger.info(
            f"PerturbationBatchSampler initialized with group_by_cell_line={group_by_cell_line}, "
            f"shuffle_batches_per_epoch={shuffle_batches_per_epoch}"
        )

        # Setup distributed settings if distributed mode is enabled.
        self.distributed = False
        self.num_replicas = 1
        self.rank = 0

        if dist.is_available() and dist.is_initialized():
            self.distributed = True
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(
                f"Distributed mode enabled. World size: {self.num_replicas}, rank: {self.rank}."
            )
            if self.shuffle_batches_per_epoch:
                logger.warning(
                    "shuffle_batches_per_epoch=True is not supported in distributed mode. "
                    "This feature will be disabled."
                )
                self.shuffle_batches_per_epoch = False

        # Create caches for all unique H5 files.
        self.metadata_caches = {}
        for subset in self.dataset.datasets:
            base_dataset: PerturbationDataset = subset.dataset
            self.metadata_caches[base_dataset.h5_path] = base_dataset.metadata_cache

        # Create batches using the code-based grouping.
        self.sentences = self._create_sentences()
        sentence_lens = [len(sentence) for sentence in self.sentences]
        avg_num = np.mean(sentence_lens)
        std_num = np.std(sentence_lens)
        tot_num = np.sum(sentence_lens)
        logger.info(
            f"Total # cells {tot_num}. Cell set size mean / std before resampling: {avg_num:.2f} / {std_num:.2f}."
        )

        # combine sentences into batches that are flattened
        logger.info(
            f"Creating meta-batches with cell_sentence_len={cell_sentence_len}..."
        )
        start_time = time.time()
        self.batches = self._create_batches()
        self.tot_num = tot_num
        end_time = time.time()
        logger.info(
            f"Sampler created with {len(self.batches)} batches in {end_time - start_time:.2f} seconds."
        )

    def _create_batches(self) -> list[list[int]]:
        """
        Combines existing batches into meta-batches of size batch_size * cell_sentence_len,
        sampling with replacement if needed to reach cell_sentence_len.

        IF distributed, each rank will process a subset of the sentences.

        If group_by_cell_line is True, batches will only contain sentences from the same cell line.
        """
        if self.distributed:
            rank_sentences = self._get_rank_sentences()
        else:
            rank_sentences = self.sentences

        # If grouping by cell line, group sentences by cell type first
        if self.group_by_cell_line:
            return self._create_batches_grouped_by_cell_line(rank_sentences)
        else:
            return self._create_batches_standard(rank_sentences)

    def _create_batches_standard(self, rank_sentences: list[list[int]]) -> list[list[int]]:
        """
        Original batch creation logic (when not grouping by cell line).
        """
        all_batches = []
        current_batch = []

        num_full = 0
        num_partial = 0
        for sentence in rank_sentences:
            # If batch is smaller than cell_sentence_len, sample with replacement
            if len(sentence) < self.cell_sentence_len and not self.test:
                # during inference, don't sample by replacement
                new_sentence = np.random.choice(
                    sentence, size=self.cell_sentence_len, replace=True
                ).tolist()
                num_partial += 1
            else:
                new_sentence = copy.deepcopy(sentence)
                assert len(new_sentence) == self.cell_sentence_len or self.test
                num_full += 1

            sentence_len = len(new_sentence) if self.test else self.cell_sentence_len

            if len(current_batch) + len(new_sentence) <= self.batch_size * sentence_len:
                current_batch.extend(new_sentence)
            else:
                if current_batch:  # Add the completed meta-batch
                    all_batches.append(current_batch)
                current_batch = new_sentence

        if self.distributed:
            logger.info(
                f"Rank {self.rank}: Of {len(rank_sentences)} sentences, {num_full} were full and {num_partial} were partial."
            )
        else:
            logger.info(
                f"Of all batches, {num_full} were full and {num_partial} were partial."
            )

        # Add the last meta-batch if it exists
        if current_batch and not self.drop_last:
            all_batches.append(current_batch)

        # Log batch statistics
        self._log_batch_statistics(all_batches)

        return all_batches

    def _create_batches_grouped_by_cell_line(self, rank_sentences: list[list[int]]) -> list[list[int]]:
        """
        Create batches grouped by cell line. Each batch contains only sentences from the same cell line.
        Sentences within each batch are shuffled randomly.
        """
        # Group sentences by cell type code
        sentences_by_cell_type: dict[int, list[list[int]]] = {}
        for sentence in rank_sentences:
            cell_type_code = self._get_cell_type_code_for_sentence(sentence)
            if cell_type_code not in sentences_by_cell_type:
                sentences_by_cell_type[cell_type_code] = []
            sentences_by_cell_type[cell_type_code].append(sentence)

        # Shuffle sentences within each cell type group
        for cell_type_code in sentences_by_cell_type:
            np.random.shuffle(sentences_by_cell_type[cell_type_code])

        all_batches = []
        num_full = 0
        num_partial = 0

        # Process each cell type separately
        for cell_type_code, cell_type_sentences in sentences_by_cell_type.items():
            current_batch = []

            for sentence in cell_type_sentences:
                # If batch is smaller than cell_sentence_len, sample with replacement
                if len(sentence) < self.cell_sentence_len and not self.test:
                    new_sentence = np.random.choice(
                        sentence, size=self.cell_sentence_len, replace=True
                    ).tolist()
                    num_partial += 1
                else:
                    new_sentence = copy.deepcopy(sentence)
                    assert len(new_sentence) == self.cell_sentence_len or self.test
                    num_full += 1

                sentence_len = len(new_sentence) if self.test else self.cell_sentence_len

                # Check if adding this sentence would exceed batch size
                if len(current_batch) + len(new_sentence) <= self.batch_size * sentence_len:
                    current_batch.extend(new_sentence)
                else:
                    # Finalize current batch if it has content
                    if current_batch:
                        # Shuffle cell indices within the batch
                        np.random.shuffle(current_batch)
                        all_batches.append(current_batch)
                    # Start new batch with this sentence
                    current_batch = new_sentence

            # Add the last batch for this cell type if it exists
            if current_batch:
                if not self.drop_last or len(current_batch) >= self.batch_size * sentence_len:
                    # Shuffle cell indices within the batch
                    np.random.shuffle(current_batch)
                    all_batches.append(current_batch)

        if self.distributed:
            logger.info(
                f"Rank {self.rank}: Created {len(all_batches)} batches grouped by cell line. "
                f"{num_full} full sentences, {num_partial} partial sentences."
            )
        else:
            logger.info(
                f"Created {len(all_batches)} batches grouped by cell line. "
                f"{num_full} full sentences, {num_partial} partial sentences."
            )

        # Log batch statistics
        self._log_batch_statistics(all_batches)

        return all_batches

    def _get_rank_sentences(self) -> list[list[int]]:
        """
        Get the subset of sentences that this rank should process.
        Sentences are shuffled using epoch-based seed, then distributed across ranks.
        """
        # Shuffle sentences using epoch-based seed for consistent ordering across ranks
        shuffled_sentences = self.sentences.copy()
        np.random.RandomState(self.seed + self.epoch).shuffle(shuffled_sentences)

        # Calculate sentence distribution across processes
        total_sentences = len(shuffled_sentences)
        base_sentences = total_sentences // self.num_replicas
        remainder = total_sentences % self.num_replicas

        # Calculate number of sentences for this specific rank
        if self.rank < remainder:
            num_sentences_for_rank = base_sentences + 1
        else:
            num_sentences_for_rank = base_sentences

        # Calculate starting sentence index for this rank
        start_sentence_idx = self.rank * base_sentences + min(self.rank, remainder)
        end_sentence_idx = start_sentence_idx + num_sentences_for_rank

        rank_sentences = shuffled_sentences[start_sentence_idx:end_sentence_idx]

        logger.info(
            f"Rank {self.rank}: Processing {len(rank_sentences)} sentences "
            f"(indices {start_sentence_idx} to {end_sentence_idx - 1} of {total_sentences})"
        )

        return rank_sentences

    def _get_cell_type_code_for_global_idx(self, global_idx: int) -> int:
        """
        Get the cell type code for a global index.

        Args:
            global_idx: Global index across all datasets

        Returns:
            Cell type code (integer)
        """
        # Find which subset this global index belongs to
        current_offset = 0
        for subset in self.dataset.datasets:
            if global_idx < current_offset + len(subset):
                # Convert global index to local index
                local_idx = subset.indices[global_idx - current_offset]
                # Get the metadata cache for this dataset
                base_dataset: PerturbationDataset = subset.dataset
                cache: H5MetadataCache = self.metadata_caches[base_dataset.h5_path]
                # Return cell type code
                return cache.cell_type_codes[local_idx]
            current_offset += len(subset)
        raise ValueError(f"Global index {global_idx} out of range")

    def _get_pert_code_for_global_idx(self, global_idx: int) -> int:
        """
        Get the perturbation code for a global index.

        Args:
            global_idx: Global index across all datasets

        Returns:
            Perturbation code (integer)
        """
        # Find which subset this global index belongs to
        current_offset = 0
        for subset in self.dataset.datasets:
            if global_idx < current_offset + len(subset):
                # Convert global index to local index
                local_idx = subset.indices[global_idx - current_offset]
                # Get the metadata cache for this dataset
                base_dataset: PerturbationDataset = subset.dataset
                cache: H5MetadataCache = self.metadata_caches[base_dataset.h5_path]
                # Return perturbation code
                return cache.pert_codes[local_idx]
            current_offset += len(subset)
        raise ValueError(f"Global index {global_idx} out of range")

    def _log_batch_statistics(self, all_batches: list[list[int]]) -> None:
        """
        Log concise batch statistics: unique perturbations per batch and batch lengths.

        Args:
            all_batches: List of batches, where each batch is a list of global indices
        """
        if not all_batches:
            return

        # Compute unique perturbations per batch
        unique_perts_per_batch = []
        batch_lengths = []

        for batch in all_batches:
            if not batch:
                continue

            batch_lengths.append(len(batch))

            # Get unique perturbation codes in this batch
            pert_codes = set()
            for global_idx in batch:
                pert_code = self._get_pert_code_for_global_idx(global_idx)
                pert_codes.add(pert_code)

            unique_perts_per_batch.append(len(pert_codes))

        if not unique_perts_per_batch:
            return

        # Compute statistics
        batch_lengths_arr = np.array(batch_lengths)
        unique_perts_arr = np.array(unique_perts_per_batch)

        batch_len_median = np.median(batch_lengths_arr)
        batch_len_min = np.min(batch_lengths_arr)
        batch_len_max = np.max(batch_lengths_arr)

        pert_median = np.median(unique_perts_arr)
        pert_min = np.min(unique_perts_arr)
        pert_max = np.max(unique_perts_arr)

        logger.info(
            f"Batch stats: lengths median/min/max = {batch_len_median:.0f}/{batch_len_min}/{batch_len_max}, "
            f"unique perts per batch median/min/max = {pert_median:.0f}/{pert_min}/{pert_max}"
        )

    def _get_cell_type_code_for_sentence(self, sentence: list[int]) -> int:
        """
        Get the cell type code for a sentence (all cells in sentence should have same cell type).

        Args:
            sentence: List of global indices

        Returns:
            Cell type code (integer)
        """
        # All cells in a sentence should have the same cell type, so just check the first one
        if not sentence:
            raise ValueError("Empty sentence")
        return self._get_cell_type_code_for_global_idx(sentence[0])

    def _process_subset(self, global_offset: int, subset: Subset) -> list[list[int]]:
        """
        Process a single subset to create batches based on H5 codes.

        Optimized version with integer group encoding:
        - Groups are encoded into a single integer via np.ravel_multi_index.
        - Sorting/grouping is done on simple integers instead of structured dtypes.
        - Much faster for large numbers of groups.
        """
        base_dataset = subset.dataset
        indices = np.array(subset.indices)
        cache: H5MetadataCache = self.metadata_caches[base_dataset.h5_path]

        # Codes
        cell_codes = cache.cell_type_codes[indices]
        pert_codes = cache.pert_codes[indices]

        if getattr(self, "use_batch", False):
            batch_codes = cache.batch_codes[indices]
            # Encode (batch, cell, pert) into one integer
            group_keys = np.ravel_multi_index(
                (batch_codes, cell_codes, pert_codes),
                (batch_codes.max() + 1, cell_codes.max() + 1, pert_codes.max() + 1),
            )
        else:
            # Encode (cell, pert) into one integer
            group_keys = np.ravel_multi_index(
                (cell_codes, pert_codes), (cell_codes.max() + 1, pert_codes.max() + 1)
            )

        # Global indices
        global_indices = np.arange(global_offset, global_offset + len(indices))

        # Sort once by group key
        order = np.argsort(group_keys)
        sorted_keys = group_keys[order]
        sorted_indices = global_indices[order]

        # Find group boundaries
        unique_keys, group_starts = np.unique(sorted_keys, return_index=True)
        group_starts = np.r_[group_starts, len(sorted_keys)]

        subset_batches = []

        # Iterate groups
        for start, end in zip(group_starts[:-1], group_starts[1:]):
            group_indices = sorted_indices[start:end]
            np.random.shuffle(group_indices)

            for i in range(0, len(group_indices), self.cell_sentence_len):
                sentence = group_indices[i : i + self.cell_sentence_len]
                if len(sentence) < self.cell_sentence_len and self.drop_last:
                    continue
                subset_batches.append(sentence.tolist())

        return subset_batches

    def _create_sentences(self) -> list[list[int]]:
        """
        Process each subset sequentially (across all datasets) and combine the batches.
        If shuffle_batches_per_epoch is True, sentences are shuffled using epoch-based seed.
        """
        global_offset = 0
        all_batches = []
        for subset in self.dataset.datasets:
            subset_batches = self._process_subset(global_offset, subset)
            all_batches.extend(subset_batches)
            global_offset += len(subset)
        
        # Shuffle sentences using epoch-based seed if shuffle_batches_per_epoch is enabled
        if self.shuffle_batches_per_epoch:
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(all_batches)
        else:
            np.random.shuffle(all_batches)

        return all_batches

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle the order of batches each time we iterate in non-distributed mode.
        if not self.distributed:
            self.batches = self._create_batches()
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        This ensures all replicas use a different random ordering for each epoch.
        If shuffle_batches_per_epoch is True, sentences are reshuffled before recreating batches.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
        # If shuffle_batches_per_epoch is enabled, reshuffle sentences before recreating batches
        if self.shuffle_batches_per_epoch:
            self.sentences = self._create_sentences()
        # Recreate batches for new epoch
        self.batches = self._create_batches()