from torch.utils.data import ConcatDataset, Dataset


class MetadataConcatDataset(ConcatDataset):
    """
    ConcatDataset that enforces consistent metadata across all constituent datasets.
    """

    def __init__(self, datasets: list[Dataset]):
        super().__init__(datasets)
        base = datasets[0].dataset
        self.embed_key = base.embed_key
        self.control_pert = base.control_pert
        self.pert_col = base.pert_col
        self.batch_col = base.batch_col

        for ds in datasets:
            md = ds.dataset
            if (
                md.embed_key != self.embed_key
                or md.control_pert != self.control_pert
                or md.pert_col != self.pert_col
                or md.batch_col != self.batch_col
            ):
                raise ValueError(
                    "All datasets must share the same embed_key, control_pert, pert_col, and batch_col"
                )
