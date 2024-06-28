import torch

import abc
import pickle

class LatentConceptSample:
    def __init__(self, sample_id, latent_concept_name, latent_concept_value, value_index):
        self.sample_id = sample_id
        self.latent_concept_name = latent_concept_name
        self.latent_concept_value = latent_concept_value
        self.latent_concept_value_index = value_index

    def to_dict(self):
        return {
            "id": self.sample_id,
            "latent_concept_name": self.latent_concept_name,
            "latent_concept_value": self.latent_concept_value,
            "latent_concept_value_index": self.latent_concept_value_index
        }

    @staticmethod
    def from_dict(latent_concept_sample_dict):
        return LatentConceptSample(latent_concept_sample_dict["id"],
                                   latent_concept_sample_dict["latent_concept_name"],
                                   latent_concept_sample_dict["latent_concept_value"],
                                   latent_concept_sample_dict["latent_concept_value_index"])


class Sample:
    def __init__(self, sample_id, raw_inputs_data, symbolic_inputs_data, target_labels, target_labels_indices):
        self.sample_id = sample_id
        self.raw_inputs_data = raw_inputs_data
        self.symbolic_inputs_data = symbolic_inputs_data
        self.target_labels = target_labels
        self.target_labels_indices = target_labels_indices

    def to_dict(self):
        return {
            "id": self.sample_id,
            "raw_inputs_data": self.raw_inputs_data,
            "symbolic_inputs_data": self.symbolic_inputs_data,
            "target_labels": self.target_labels,
            "target_labels_indices": self.target_labels_indices
        }

    @staticmethod
    def from_dict(sample_dict):
        return Sample(sample_dict["id"], sample_dict["raw_inputs_data"], sample_dict["symbolic_inputs_data"],
                      sample_dict["target_labels"], sample_dict["target_labels_indices"])



class LatentConceptDataset(torch.utils.data.Dataset):
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self, idx, split):
        raise NotImplementedError()


class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, task_metadata, task_data_filepath, latent_concept_datasets, split):
        super().__init__()
        assert split in ["train", "val", "test"]
        self._split = split
        self._metadata = task_metadata
        self._latent_concept_datasets = latent_concept_datasets[split]
        self._samples = self._load_data(task_data_filepath)

    def _load_data(self, task_data_path):
        with open(task_data_path, 'rb') as task_data_fp:
            task_data = pickle.load(task_data_fp)[self._split]
        return [Sample.from_dict(sample_dict) for sample_dict in task_data]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        sample = self._samples[idx]

        raw_inputs_and_labels = [
            self._latent_concept_datasets[self._metadata.raw_inputs[raw_input_name].concept_name][raw_sample_idx]
            for raw_input_name, raw_sample_idx in sample.raw_inputs_data.items()
        ]
        raw_inputs = [
            inputs_and_labels[0]
            for inputs_and_labels in raw_inputs_and_labels
        ]

        symbolic_inputs = [
            torch.tensor(symbolic_input_data["value_idx"])
            for symbolic_input_data in sample.symbolic_inputs_data.values()
        ]

        latent_labels_num_classes = [
            len(self._metadata.latent_concepts[self._metadata.raw_inputs[raw_input_name].concept_name].values)
            for raw_input_name in sample.raw_inputs_data.keys()
        ]
        latent_labels = [
            torch.nn.functional.one_hot(inputs_and_labels[1], num_classes=num_classes).float()
            for inputs_and_labels, num_classes in zip(raw_inputs_and_labels, latent_labels_num_classes)
        ]

        target_labels = list()
        for target_name, target_data in sample.target_labels_indices.items():
            target_size = len(self._metadata.target_concepts[self._metadata.targets[target_name].target_concept_name].values)
            target_tensor = torch.zeros(size=(target_size,), dtype=torch.float32)
            target_tensor[target_data] = 1.0
            target_labels.append(target_tensor)

        return raw_inputs, symbolic_inputs, latent_labels, target_labels


