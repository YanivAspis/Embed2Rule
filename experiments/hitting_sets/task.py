from experiments.reasoning_networks import ResBlockReasoning
from experiments.perception_networks import LeNet
from experiments.raw_datasets import MNISTDataset

def get_latent_concept_datasets(config, split, additional_args):
    return {
        "element": MNISTDataset(split),
    }


def get_networks(config, metadata):
    perception_network = LeNet(config["model"]["embedding_size"])
    perception_networks = {
        "element": perception_network
    }
    reasoning_network = ResBlockReasoning(config, metadata, {
        "element": config["model"]["embedding_size"]
    })
    return perception_networks, reasoning_network