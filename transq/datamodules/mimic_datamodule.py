from transq.datasets import MIMIC_Dataset
from .datamodule_mrg import BaseDataModule


class MIMICDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MIMIC_Dataset

    @property
    def dataset_name(self):
        return "mimic"
