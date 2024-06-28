import torch
import numpy as np
from munkres import Munkres, print_matrix

import pickle
from os import path

from framework.dataset import TaskDataset

class ClusterLabeller:
    def __init__(self, config, metadata):
        self._config = config
        self._metadata = metadata
        self._setup_index_permutations()

    def _setup_index_permutations(self):
        self._index_permutations = {
            latent_concept_name: np.empty((len(concept.values), ))
            for latent_concept_name, concept in self._metadata.latent_concepts.items()
        }

    def set_permutation(self, concept_name, permutation):
        self._index_permutations[concept_name] = permutation

    def label_for_concept(self, concept_name, cluster_indices):
        return np.array([self._index_permutations[concept_name][cluster_index] for cluster_index in cluster_indices])

    def label(self, cluster_index_list):
        return [
            self.label_for_concept(raw_input.concept_name, cluster_indices)
            for raw_input, cluster_indices in zip(self._metadata.raw_inputs.values(), cluster_index_list)
        ]

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(self._index_permutations, f)

    def load(self, load_path):
        with open(load_path, "rb") as f:
            self._index_permutations = pickle.load(f)


class ClusterLabelOptimiser:
    def __init__(self, config, device, metadata, cluster_labeller, neural_model, clustering, weak_labeller, data_path, latent_concept_datasets):
        self._config = config
        self._device = device
        self._metadata = metadata
        self._cluster_labeller = cluster_labeller
        self._clustering = clustering
        self._neural_model = neural_model
        self._weak_labeller = weak_labeller
        self._setup_dataloaders(data_path, latent_concept_datasets)

    def _setup_dataloaders(self, data_path, latent_concept_datasets):
        self._train_loader = torch.utils.data.DataLoader(
            TaskDataset(self._metadata, data_path, latent_concept_datasets, "train"),
            batch_size=self._config["weak_labelling"]["batch_size"],
            shuffle=True,
            drop_last=True
        )

    @torch.no_grad()
    def _get_data(self):
        data = {
            latent_concept_name: {
                "cluster_indices": list(),
                "weak_labels": list()
            }
            for latent_concept_name in self._metadata.latent_concepts.keys()
        }
        for batch_raw_inputs, batch_symbolic_inputs, _, _ in self._train_loader:
            for batch_raw_input, raw_input in zip(batch_raw_inputs, self._metadata.raw_inputs.values()):
                concept_name = raw_input.concept_name
                if sum([tensor.shape[0] for tensor in data[concept_name]["weak_labels"]]) >= self._config["weak_labelling"]["num_samples"]:
                    continue
                probs = self._weak_labeller.classify_images(batch_raw_input, concept_name)
                data[concept_name]["weak_labels"].append(torch.argmax(probs, dim=-1).cpu().numpy())
            raw_inputs = [raw_input.to(self._device) for raw_input in batch_raw_inputs]
            symbolic_inputs = [symbolic_input.to(self._device) for symbolic_input in batch_symbolic_inputs]
            embeddings = self._neural_model(raw_inputs, symbolic_inputs, return_embeddings=True)
            embeddings = [embedding.cpu().numpy() for embedding in embeddings]
            cluster_index_list = self._clustering.predict(embeddings)
            for cluster_indices, raw_input in zip(cluster_index_list, self._metadata.raw_inputs.values()):
                concept_name = raw_input.concept_name
                if sum([tensor.shape[0] for tensor in data[concept_name]["cluster_indices"]]) >= self._config["weak_labelling"]["num_samples"]:
                    continue
                data[concept_name]["cluster_indices"].append(cluster_indices)
            num_samples_loaded = max([len(data[latent_concept_name]["cluster_indices"]) for latent_concept_name in data.keys()])
            if num_samples_loaded >= self._config["weak_labelling"]["num_samples"]:
                break
        data = {
            latent_concept_name: {
                "cluster_indices": np.concatenate(data[latent_concept_name]["cluster_indices"], axis=0),
                "weak_labels": np.concatenate(data[latent_concept_name]["weak_labels"], axis=0)
            }
            for latent_concept_name in data.keys()
        }
        return data

    def optimise_cluster_labels(self, results_dir):
        data = self._get_data()
        for latent_concept_name in self._metadata.latent_concepts.keys():
            self._optimise_concept_cluster_labels(latent_concept_name, data[latent_concept_name])
        self._cluster_labeller.save(path.join(results_dir, "cluster_labels.pkl"))

    def _optimise_concept_cluster_labels(self, concept_name, data):
        permutation = self._get_best_permutation(concept_name, data["cluster_indices"], data["weak_labels"])
        self._cluster_labeller.set_permutation(concept_name, permutation)


    def _get_best_permutation(self, concept_name, cluster_indices, weak_labels):
        num_clusters = len(self._metadata.latent_concepts[concept_name].values)
        assignment_matrix = np.zeros(shape=(num_clusters, num_clusters), dtype=np.int32)
        for cluster_idx, label in zip(cluster_indices, weak_labels):
            assignment_matrix[cluster_idx][label] += 1

        verbose_mode = "labelling" in self._config and "verbose" in self._config["labelling"] and self._config["labelling"]["verbose"]

        m = Munkres()
        indexes = m.compute(-assignment_matrix)
        if verbose_mode:
            print("Optimisation info for concept", concept_name)
            print_matrix(assignment_matrix, msg='Highest utility through this matrix:')
        total = 0
        assignment = np.empty(shape=(len(indexes),), dtype=np.int32)
        for row, column in indexes:
            value = assignment_matrix[row][column]
            total += value
            if verbose_mode:
                print(f'({row}, {column}) -> {value}')
            assignment[row] = column
        if verbose_mode:
            print(f'total utility: {total}')
        return assignment

def load_cluster_labeller(cluster_labeller, results_dir):
    cluster_labeller.load(path.join(results_dir, "cluster_labels.pkl"))
    return cluster_labeller