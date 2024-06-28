import torch
import numpy as np

from framework.dataset import TaskDataset
from framework.asp_handler import ASPHandler
from experiments.metrics import categorical_accuracy, binary_accuracy

class NeuroSymbolicModel:
    def __init__(self, config, device, metadata, fully_neural, clustering, cluster_labeller, background_file, hypothesis_file):
        self._config = config
        self._device = device
        self._metadata = metadata
        self._fully_neural = fully_neural.to(device)
        self._setup_metrics(metadata)
        self._clustering = clustering
        self._cluster_labeller = cluster_labeller
        self._asp_handler = ASPHandler(config, metadata, background_file, hypothesis_file)

    def _setup_metrics(self, metadata):
        self._metrics = dict()
        for target_name, target_data in metadata.targets.items():
            if target_data.multi_valued:
                self._metrics[target_name] = binary_accuracy
            else:
                self._metrics[target_name] = categorical_accuracy

    @torch.no_grad()
    def predict_latent(self, raw_inputs):
        raw_inputs = [raw_input.to(self._device) for raw_input in raw_inputs]
        embeddings = self._fully_neural(raw_inputs, [], return_embeddings=True)
        embeddings = [embedding.cpu().numpy() for embedding in embeddings]
        cluster_indices = self._clustering.predict(embeddings)
        latent_predictions = self._cluster_labeller.label(cluster_indices)
        return latent_predictions

    def to_zero_one_array(self, target_name, target_indices):
        target_size = len(self._metadata.target_concepts[self._metadata.targets[target_name].target_concept_name].values)
        target_tensor = np.identity(target_size, dtype=np.float32)[target_indices][:,0]
        return target_tensor

    @torch.no_grad()
    def predict(self, raw_inputs, symbolic_inputs):
        latent_predictions = self.predict_latent(raw_inputs)
        symbolic_inputs = [symbolic_input.cpu().numpy() for symbolic_input in symbolic_inputs]

        target_predictions = list()
        for sample_idx in range(raw_inputs[0].shape[0]):
            ex_latent_predictions = [latent_prediction[sample_idx] for latent_prediction in latent_predictions]
            ex_symbolic_inputs = [symbolic_input[sample_idx] for symbolic_input in symbolic_inputs]
            target_predictions.append(self._asp_handler(ex_latent_predictions, ex_symbolic_inputs))

        target_predictions = [
            self.to_zero_one_array(target_name, np.array([target_predictions[sample_idx][target_idx]
                                                          for sample_idx in range(len(target_predictions))]))
            for target_idx, target_name in enumerate(self._metadata.targets.keys())
        ]
        return target_predictions

    @torch.no_grad()
    def evaluate_fully_neural(self, data_path, split, latent_concept_datasets):
        dataloader = torch.utils.data.DataLoader(
            TaskDataset(self._metadata, data_path, latent_concept_datasets, split),
            batch_size=self._config["train"]["batch_size"],
            shuffle=False
        )

        results = {
            target_name: list()
            for target_name in self._metadata.targets.keys()
        }

        for raw_inputs, symbolic_inputs, _, target_labels in dataloader:
            raw_inputs = [raw_input.to(self._device) for raw_input in raw_inputs]
            symbolic_inputs = [symbolic_input.to(self._device) for symbolic_input in symbolic_inputs]
            target_labels = [labels.to(self._device) for labels in target_labels]
            target_predictions = self._fully_neural(raw_inputs, symbolic_inputs)
            for target_name, prediction, target_label in zip(self._metadata.targets.keys(), target_predictions, target_labels):
                results[target_name].append(self._metrics[target_name](prediction, target_label).cpu().numpy())

        return {
            target_name: float(np.mean(result))
            for target_name, result in results.items()
        }

    @torch.no_grad()
    def evaluate_latent(self, data_path, split, latent_concept_datasets):
        dataloader = torch.utils.data.DataLoader(
            TaskDataset(self._metadata, data_path, latent_concept_datasets, split),
            batch_size=self._config["train"]["batch_size"],
            shuffle=False
        )

        results = {
            latent_concept_name: {
                "correct": 0,
                "total": 0
            }
            for latent_concept_name in self._metadata.latent_concepts.keys()
        }

        for raw_inputs, symbolic_inputs, latent_labels, _ in dataloader:
            latent_labels = [torch.argmax(labels, dim=-1).cpu().numpy() for labels in latent_labels]
            latent_predictions = self.predict_latent(raw_inputs)
            for raw_input, prediction, latent_label in zip(self._metadata.raw_inputs.values(), latent_predictions, latent_labels):
                concept_name = raw_input.concept_name
                results[concept_name]["correct"] += np.sum(prediction == latent_label)
                results[concept_name]["total"] += len(prediction)

        return {
            latent_concept_name: result["correct"] / result["total"]
            for latent_concept_name, result in results.items()
        }

    def evaluate_neurosymbolic(self, data_path, split, latent_concept_datasets):
        dataloader = torch.utils.data.DataLoader(
            TaskDataset(self._metadata, data_path, latent_concept_datasets, split),
            batch_size=self._config["train"]["batch_size"],
            shuffle=False
        )

        results = {
            target_name: {
                "correct": 0,
                "total": 0
            }
            for target_name in self._metadata.targets.keys()
        }

        for raw_inputs, symbolic_inputs, _, target_labels in dataloader:
            target_labels = [labels.cpu().numpy() for labels in target_labels]
            target_predictions = self.predict(raw_inputs, symbolic_inputs)
            for target_name, prediction, target_label in zip(self._metadata.targets.keys(), target_predictions, target_labels):
                results[target_name]["correct"] += np.sum(np.all(prediction == target_label, axis=1))
                results[target_name]["total"] += len(prediction)

        return {
            target_name: result["correct"] / result["total"]
            for target_name, result in results.items()
        }