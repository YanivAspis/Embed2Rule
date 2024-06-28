import torch

from os import path

from framework.dataset import TaskDataset
from framework.ilasp_task import ILASPTask, ILASPExample
from utils import normalise_asp_constant

class SymbolicLearner:
    def __init__(self, config, device, metadata, model, clustering, cluster_labeller, experiment_dir, latent_concept_datasets):
        self._config = config
        self._device = device
        self._metadata = metadata
        self._model = model
        self._clustering = clustering
        self._cluster_labeller = cluster_labeller
        self._background_file = path.join(experiment_dir, "background.lp")
        self._mode_bias_file = path.join(experiment_dir, "mode_bias.lp")
        self._setup_dataloader(path.join(experiment_dir, "data.pkl"), latent_concept_datasets)

    def _setup_dataloader(self, data_path, latent_concept_datasets):
        self._dataloader = torch.utils.data.DataLoader(
            dataset=TaskDataset(
                self._metadata,
                data_path,
                latent_concept_datasets,
                "train"
            ),
            batch_size=self._config["train"]["batch_size"],
            shuffle=True,
            num_workers=0
        )

    def _generate_example(self, ex_idx, latent_predictions, symbolic_inputs, targets_indices):
        inclusion_set = list()
        exclusion_set = list()
        context = list()
        is_pos = True

        if self._config["symbolic_learner"]["learning_mode"] == "classification":
            is_pos = True
            for target_name, target_indices in zip(self._metadata.targets.keys(), targets_indices):
                target_concept_name = self._metadata.targets[target_name].target_concept_name
                num_target_values = len(self._metadata.target_concepts[target_concept_name].values)
                for target_idx in range(num_target_values):
                    target_value = self._metadata.target_concepts[target_concept_name].values[target_idx]
                    if target_idx in target_indices:
                        inclusion_set.append(f"{target_name}({target_value})")
                    else:
                        exclusion_set.append(f"{target_name}({target_value})")
        elif self._config["symbolic_learner"]["learning_mode"] == "existence":
            target_concept_name = next(iter(self._metadata.targets.values())).target_concept_name
            if self._metadata.target_concepts[target_concept_name].values[targets_indices[0]] == "true":
                is_pos = True
            else:
                is_pos = False

        for raw_input_name, raw_input_index in zip(self._metadata.raw_inputs.keys(), latent_predictions):
            concept_name = self._metadata.raw_inputs[raw_input_name].concept_name
            concept_val = self._metadata.latent_concepts[concept_name].structured_values[raw_input_index]
            context.append(f"holds({raw_input_name}, {normalise_asp_constant(concept_val)}).")

        for symbolic_input_name, symbolic_input_index in zip(self._metadata.symbolic_inputs.keys(), symbolic_inputs):
            concept_name = self._metadata.symbolic_inputs[symbolic_input_name].concept_name
            concept_val = self._metadata.symbolic_concepts[concept_name].structured_values[symbolic_input_index]
            context.append(f"holds({symbolic_input_name}, {normalise_asp_constant(concept_val)}).")

        return ILASPExample(
            is_pos,
            ex_idx,
            self._config["symbolic_learner"]["example_weight"],
            inclusion_set,
            exclusion_set,
            context
        )

    @torch.no_grad()
    def _generate_examples(self):
        examples = list()
        ex_idx = 0
        for raw_inputs, symbolic_inputs, _, target_labels in self._dataloader:
            raw_inputs = [raw_input.to(self._device) for raw_input in raw_inputs]
            embeddings = self._model(raw_inputs, [], return_embeddings=True)
            embeddings = [embedding.cpu().numpy() for embedding in embeddings]
            cluster_indices = self._clustering.predict(embeddings)
            latent_predictions = self._cluster_labeller.label(cluster_indices)
            targets_indices = [torch.argmax(target_label, dim=1) for target_label in target_labels]

            for i in range(raw_inputs[0].shape[0]):
                ex_latent_predictions = [latent_prediction[i] for latent_prediction in latent_predictions]
                ex_symbolic_indices = [symbolic_index[i] for symbolic_index in symbolic_inputs]
                ex_targets_indices = [target_index[i] for target_index in targets_indices]
                examples.append(self._generate_example(ex_idx, ex_latent_predictions, ex_symbolic_indices, ex_targets_indices))
                ex_idx += 1
                if len(examples) >= self._config["symbolic_learner"]["num_examples"]:
                    break

            if len(examples) >= self._config["symbolic_learner"]["num_examples"]:
                break

        return examples


    def learn_hypothesis(self, results_dir):
        return ILASPTask(
            self._config,
            self._background_file,
            self._mode_bias_file,
            self._generate_examples()
        )(results_dir)