import hydra
from omegaconf import DictConfig

CONFIG_PATH = "../../config"


@hydra.main(config_path=CONFIG_PATH, config_name="main", version_base=None)
def train_model(config: DictConfig):
    """
    Trains a model using the provided configurations.
    """

    print(f"Train modeling using {config.data.interim}")
    print(f"Model used: {config.model.name}")
    print(f"Model parameters: {config.model.parameters}")
    print(f"Save the output to {config.data.processed}")


if __name__ == "__main__":
    train_model()
