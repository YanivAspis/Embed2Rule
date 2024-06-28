from experiments.transformer import LatentConceptTransformer
from experiments.perception_networks import DigitConv
from experiments.raw_datasets import MNISTDataset

def get_latent_concept_datasets(config, split, additional_args):
    return {
        "element": MNISTDataset(split),
    }


def get_networks(config, metadata):
    perception_network = DigitConv(config["model"]["embedding_size"])
    perception_networks = {
        "element": perception_network
    }
    
    reasoning_network = LatentConceptTransformer(metadata, config["model"]["reasoning_output_size"], {
        "element": config["model"]["embedding_size"]
    })
    return perception_networks, reasoning_network