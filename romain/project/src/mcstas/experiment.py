from dataclasses import dataclass
from pathlib import Path

@dataclass
class Experiment:
    name: str
    scan_point: dict
    folder: Path

    @property
    def mat(self):
        return self.folder / "mat"

    @property
    def smat(self):
        return self.folder / "smat"