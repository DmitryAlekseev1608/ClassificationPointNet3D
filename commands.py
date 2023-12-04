from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class DatasetConfig:
    path: str = "data/ModelNet10"

cs = ConfigStore.instance()
cs.store(name="config_set", node=DatasetConfig)


if __name__ == "__main__":
    my_app()