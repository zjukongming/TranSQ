from .mimic_datamodule import MIMICDataModule
from .iuxray_datamodule import IUXRAYDataModule

_datamodules = {
    "mimic": MIMICDataModule,
    "iuxray": IUXRAYDataModule,
}
