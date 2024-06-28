import json


class Concept:
    def __init__(self, name, values, structured_values=None):
        self.name = name
        self.values = values
        if structured_values is None:
            self.structured_values = values
        else:
            self.structured_values = structured_values

    def to_dict(self):
        return {
            "name": self.name,
            "values": self.values,
            "structured_values": self.structured_values
        }

    @staticmethod
    def from_dict(sample_dict):
        return Concept(sample_dict["name"], sample_dict["values"], sample_dict["structured_values"])


class TaskInput:
    def __init__(self, name, concept_name):
        self.name = name
        self.concept_name = concept_name

    def to_dict(self):
        return {
            "name": self.name,
            "concept_name": self.concept_name
        }

    @staticmethod
    def from_dict(raw_input_dict):
        return TaskInput(raw_input_dict["name"], raw_input_dict["concept_name"])


class Target:
    def __init__(self, name, target_concept_name, multi_valued):
        self.name = name
        self.target_concept_name = target_concept_name
        self.multi_valued = multi_valued

    def to_dict(self):
        return {
            "name": self.name,
            "target_concept_name": self.target_concept_name,
            "multi_valued": self.multi_valued
        }

    @staticmethod
    def from_dict(target_dict):
        return Target(target_dict["name"], target_dict["target_concept_name"], target_dict["multi_valued"])



class TaskMetadata:
    def __init__(self):
        self.raw_inputs = dict()
        self.symbolic_inputs = dict()
        self.targets = dict()
        self.latent_concepts = dict()
        self.symbolic_concepts = dict()
        self.target_concepts = dict()
        self.inputs_to_concepts = dict()
        self.target_to_target_concepts = dict()

    def add_latent_concept(self, concept):
        self.latent_concepts[concept.name] = concept

    def add_symbolic_concept(self, concept):
        self.symbolic_concepts[concept.name] = concept

    def add_target_concept(self, target_concept):
        self.target_concepts[target_concept.name] = target_concept

    def add_raw_input(self, raw_input):
        self.raw_inputs[raw_input.name] = raw_input
        self.inputs_to_concepts[raw_input.name] = raw_input.concept_name

    def add_symbolic_input(self, symbolic_input):
        self.symbolic_inputs[symbolic_input.name] = symbolic_input
        self.inputs_to_concepts[symbolic_input.name] = symbolic_input.concept_name

    def add_target(self, target):
        self.targets[target.name] = target
        self.target_to_target_concepts[target.name] = target.target_concept_name

    def to_dict(self):
        return {
            "latent_concepts": [latent_concept.to_dict() for latent_concept in self.latent_concepts.values()],
            "symbolic_concepts": [symbolic_concept.to_dict() for symbolic_concept in self.symbolic_concepts.values()],
            "target_concepts": [target_concept.to_dict() for target_concept in self.target_concepts.values()],
            "raw_inputs": [raw_input.to_dict() for raw_input in self.raw_inputs.values()],
            "symbolic_inputs": [symbolic_input.to_dict() for symbolic_input in self.symbolic_inputs.values()],
            "targets": [target.to_dict() for target in self.targets.values()],
        }

    @staticmethod
    def from_dict(metadata_dict):
        metadata = TaskMetadata()
        for latent_concept in metadata_dict["latent_concepts"]:
            metadata.add_latent_concept(Concept.from_dict(latent_concept))
        for symbolic_concept in metadata_dict["symbolic_concepts"]:
            metadata.add_symbolic_concept(Concept.from_dict(symbolic_concept))
        for target_concept in metadata_dict["target_concepts"]:
            metadata.add_target_concept(Concept.from_dict(target_concept))
        for raw_input in metadata_dict["raw_inputs"]:
            metadata.add_raw_input(TaskInput.from_dict(raw_input))
        for symbolic_input in metadata_dict["symbolic_inputs"]:
            metadata.add_symbolic_input(TaskInput.from_dict(symbolic_input))
        for target in metadata_dict["targets"]:
            metadata.add_target(Target.from_dict(target))
        return metadata

    def to_json(self, json_filepath):
        with open(json_filepath, 'w') as json_fp:
            json.dump(self.to_dict(), json_fp)

    @staticmethod
    def from_json(json_filepath):
        with open(json_filepath, 'r') as json_fp:
            return TaskMetadata.from_dict(json.load(json_fp))