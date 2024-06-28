import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from os import path, makedirs

from framework.dataset import TaskDataset


class TSNEPlotGenerator:
    def __init__(self, config, device, metadata, data_path, latent_concept_datasets, split, results_dir):
        self._config = config
        self._device = device
        self._metadata = metadata
        self._setup_dataloader(data_path, latent_concept_datasets, split)
        self._plots_dir = path.join(results_dir, "latent_plots", split)

    def _setup_dataloader(self, data_path, latent_concept_datasets, split):
        self._dataloader = torch.utils.data.DataLoader(
            TaskDataset(self._metadata, data_path, latent_concept_datasets, split),
            batch_size=self._config["train"]["batch_size"],
            shuffle=True
        )

    @torch.no_grad()
    def _get_data(self, model):
        data = {
            latent_concept_name: {
                "embeddings": list(),
                "labels": list()
            }
            for latent_concept_name in self._metadata.latent_concepts.keys()
        }
        num_points = {
            latent_concept_name: 0
            for latent_concept_name in self._metadata.latent_concepts.keys()
        }
        for raw_inputs, symbolic_inputs, latent_labels, _ in self._dataloader:
            raw_inputs = [raw_input.to(self._device) for raw_input in raw_inputs]
            symbolic_inputs = [symbolic_input.to(self._device) for symbolic_input in symbolic_inputs]
            latent_concept_embeddings = model(raw_inputs, symbolic_inputs, return_embeddings=True)
            for raw_input_name, latent_concept_embedding, latent_gt in zip(self._metadata.raw_inputs.keys(), latent_concept_embeddings, latent_labels):
                if num_points[self._metadata.raw_inputs[raw_input_name].concept_name] < self._config["visualisation"]["num_points"]:
                    data[self._metadata.raw_inputs[raw_input_name].concept_name]["embeddings"].append(latent_concept_embedding.cpu().numpy())
                    data[self._metadata.raw_inputs[raw_input_name].concept_name]["labels"].append(np.argmax(latent_gt.cpu().numpy(), axis=-1))
                    num_points[self._metadata.raw_inputs[raw_input_name].concept_name] += latent_concept_embedding.shape[0]
            if min([num_points[latent_concept_name] for latent_concept_name in data.keys()]) >= self._config["visualisation"]["num_points"]:
                break
        data = {
            latent_concept_name: {
                "embeddings": np.concatenate(data[latent_concept_name]["embeddings"], axis=0),
                "labels": np.concatenate(data[latent_concept_name]["labels"], axis=0),
            }
            for latent_concept_name, latent_data in data.items()
        }
        return data

    def _get_tsne_values(self, embeddings):
        reduced_embeddings = TSNE(n_components=self._config["visualisation"]["n_components"],
                                  learning_rate=self._config["visualisation"]["learning_rate"],
                                  init=self._config["visualisation"]["init"],
                                  perplexity=self._config["visualisation"]["perplexity"]).fit_transform(embeddings)
        return reduced_embeddings

    def _plot(self, latent_concept_name, data, epoch):
        labels = self._metadata.latent_concepts[latent_concept_name].values

        label_to_points = {
            label: list()
            for label in labels
        }
        for point, label in zip(data["points"], data["labels"]):
            label_to_points[labels[label]].append(point)

        NUM_COLORS = len(labels)
        cm = plt.get_cmap('hsv')
        colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]

        axs = list()
        for label, points in label_to_points.items():
            points = np.array(points)
            axs.append(plt.scatter(points[:, 0], points[:, 1], label=label, color=colors[labels.index(label)]))
        plt.legend(axs, self._metadata.latent_concepts[latent_concept_name].values)

        save_dir = path.join(self._plots_dir, latent_concept_name)
        makedirs(save_dir, exist_ok=True)
        plt.savefig(path.join(save_dir, f"epoch_{epoch}.png"))
        plt.close()

    @torch.no_grad()
    def __call__(self, model, epoch):
        data = self._get_data(model)
        data = {
            latent_concept_name: {
                "points": self._get_tsne_values(data[latent_concept_name]["embeddings"]),
                "labels": data[latent_concept_name]["labels"]
            }
            for latent_concept_name in data.keys()
        }
        for latent_concept_name in data.keys():
            self._plot(latent_concept_name, data[latent_concept_name], epoch)


