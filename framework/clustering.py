import numpy as np
import torch
from sklearn.cluster import KMeans

from os import path
import pickle

from framework.dataset import TaskDataset

class Clustering:
    def __init__(self, metadata):
        self._metadata = metadata
        self._setup_clusters()

    def _setup_clusters(self):
        self._concept_to_clusters = {
            concept_name: KMeans(n_clusters=len(concept.values), n_init="auto")
            for concept_name, concept in self._metadata.latent_concepts.items()
        }

    def _train_cluster(self, concept_name, data):
        self._concept_to_clusters[concept_name].fit(data)

    def train(self, data):
        for latent_concept_name in self._metadata.latent_concepts.keys():
            self._train_cluster(latent_concept_name, data[latent_concept_name])

    def _predict_cluster(self, concept_name, data):
        return self._concept_to_clusters[concept_name].predict(data)

    def predict(self, cluster_index_list):
        return [
            self._predict_cluster(raw_input.concept_name, cluster_indices)
            for raw_input, cluster_indices in zip(self._metadata.raw_inputs.values(), cluster_index_list)
        ]

    def predict_for_concept(self, concept_name, cluster_indices):
        return self._concept_to_clusters[concept_name].predict(cluster_indices)

    def save_clustering(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(self._concept_to_clusters, f)

    def load_clustering(self, load_path):
        with open(load_path, "rb") as f:
            self._concept_to_clusters = pickle.load(f)

class ClusterTrainer:
    def __init__(self, config, device, metadata, clustering, perception_networks, data_path, latent_concept_datasets):
        self._config = config
        self._device = device
        self._metadata = metadata
        self._perception_networks = perception_networks
        self._clustering = clustering
        self._setup_dataloaders(data_path, latent_concept_datasets)

    def _setup_dataloaders(self, data_path, latent_concept_datasets):
        self._train_loader = torch.utils.data.DataLoader(
            TaskDataset(self._metadata, data_path, latent_concept_datasets, "train"),
            batch_size=self._config["train"]["batch_size"],
            shuffle=True
        )
        self._valid_loader = torch.utils.data.DataLoader(
            TaskDataset(self._metadata, data_path, latent_concept_datasets, "val"),
            batch_size=self._config["train"]["batch_size"],
            shuffle=False
        )

    def _load_data(self):
        num_samples_loaded = 0
        data = {
            raw_input.concept_name: list()
            for raw_input in self._metadata.raw_inputs.values()
        }
        for raw_inputs_data, _, _, _ in self._train_loader:
            for raw_input, raw_input_data in zip(self._metadata.raw_inputs.values(), raw_inputs_data):
                raw_input_data = raw_input_data.to(self._device)
                raw_input_embeddings = self._perception_networks[raw_input.concept_name](raw_input_data)
                data[raw_input.concept_name].append(raw_input_embeddings.cpu().numpy())
            num_samples_loaded += len(raw_inputs_data[0])
            if num_samples_loaded >= self._config["clustering"]["num_train_samples"]:
                break
        data = {
            raw_input.concept_name: np.concatenate(data[raw_input.concept_name], axis=0)
            for raw_input in self._metadata.raw_inputs.values()
        }
        return data

    @torch.no_grad()
    def train(self, results_dir):
        for network in self._perception_networks.values():
            network.eval()
        self._clustering.train(self._load_data())
        self.save_clusters(results_dir)

    def save_clusters(self, results_dir):
        self._clustering.save_clustering(path.join(results_dir, "clustering.pkl"))

def load_clusters(clustering, results_dir):
    clustering.load_clustering(path.join(results_dir, "clustering.pkl"))
    return clustering