import hydra
from omegaconf import DictConfig

CONFIG_PATH = "../../config"


@hydra.main(config_path=CONFIG_PATH, config_name="main", version_base=None)
def process_data(config: DictConfig):
    """
    Process the data using the provided configurations.
    """

    print(f"Process data using {config.data.raw}")
    print(f"Columns used: {config.process.use_columns}")


if __name__ == "__main__":
    process_data()
