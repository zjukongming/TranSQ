from transq.datasets import IUXRAY_Dataset
from .datamodule_mrg import BaseDataModule


class IUXRAYDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return IUXRAY_Dataset

    @property
    def dataset_name(self):
        return "iuxray"
