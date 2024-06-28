import torch
import torchvision

from experiments.reasoning_networks import ResBlockReasoning
from experiments.raw_datasets import CardsDataset

def get_latent_concept_datasets(config, split, additional_args = None):
    return {
        "card": CardsDataset(config, additional_args["cards_data_path"], split, label_mode="all")
    }


def get_networks(config, metadata):
    perception_network = torchvision.models.resnet18()
    perception_network.fc = torch.nn.Linear(in_features=perception_network.fc.in_features,
                                            out_features=config["model"]["embedding_size"])
    perception_networks = {
        "card": perception_network
    }
    reasoning_network = ResBlockReasoning(config, metadata, {
        "card": config["model"]["embedding_size"]
    })
    return perception_networks, reasoning_network