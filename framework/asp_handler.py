import numpy as np
import clingo

from utils import normalise_asp_constant

class ASPHandler:
    def __init__(self, config, metadata, background_file, hypothesis_file):
        self._config = config
        self._metadata = metadata
        self._setup_program(background_file, hypothesis_file)

    def _setup_program(self, background_file, hypothesis_file):
        with open(background_file, "r") as f:
            self._program = f.read()
        self._program += "\n"
        with open(hypothesis_file, "r") as f:
            self._program += f.read()

    def _latent_index_to_atom(self, raw_input_name, latent_prediction):
        concept_name = self._metadata.raw_inputs[raw_input_name].concept_name
        concept_val = self._metadata.latent_concepts[concept_name].structured_values[latent_prediction]
        return f"holds({raw_input_name}, {normalise_asp_constant(concept_val)})."

    def _symbolic_index_to_atom(self, symbolic_input_name, symbolic_index):
        concept_name = self._metadata.symbolic_inputs[symbolic_input_name].concept_name
        concept_val = self._metadata.symbolic_concepts[concept_name].structured_values[symbolic_index]
        return f"holds({symbolic_input_name}, {normalise_asp_constant(concept_val)})."

    def _target_atom_to_indices(self, target_name, target_atoms):
        target_concept_name = self._metadata.targets[target_name].target_concept_name
        target_concept_values = self._metadata.target_concepts[target_concept_name].values
        target_indices = list()
        for target_atom in target_atoms:
            if target_atom.arguments[0].type == clingo.SymbolType.Number:
                target_val = str(target_atom.arguments[0].number)
            else:
                target_val = target_atom.arguments[0].name
            target_indices.append(target_concept_values.index(target_val))
        return sorted(target_indices)

    def _execute_program(self, program):
        ctl = clingo.Control()
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        with ctl.solve(yield_=True) as hnd:
            for m in hnd:
                model = m.symbols(atoms=True)
                break
        return model

    def _extract_target_atoms(self, atoms):
        target_atoms = {
            target_name: list()
            for target_name in self._metadata.targets.keys()
        }
        for atom in atoms:
            if atom.name in self._metadata.targets.keys():
                target_atoms[atom.name].append(atom)
        target_atoms = [target_atoms[target_name] for target_name in self._metadata.targets.keys()]
        return target_atoms

    def _check_answer_set_existence(self, program):
        ctl = clingo.Control()
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        with ctl.solve(yield_=True) as hnd:
            for _ in hnd:
                return True
        return False

    def __call__(self, latent_predictions, symbolic_inputs):
        atoms = list()
        for raw_input_name, latent_prediction in zip(self._metadata.raw_inputs.keys(), latent_predictions):
            atoms.append(self._latent_index_to_atom(raw_input_name, latent_prediction))
        for symbolic_input_name, symbolic_index in zip(self._metadata.symbolic_inputs.keys(), symbolic_inputs):
            atoms.append(self._symbolic_index_to_atom(symbolic_input_name, symbolic_index))
        program = self._program + "\n" + "\n".join(atoms)
        if self._config["symbolic_learner"]["learning_mode"] == "classification":
            answer_set = self._execute_program(program)
            target_atoms = self._extract_target_atoms(answer_set)
            target_predictions = list()
            for target_name, target_atom in zip(self._metadata.targets.keys(), target_atoms):
                target_predictions.append(np.array(self._target_atom_to_indices(target_name, target_atom)))
        else:
            answer_set_exists = self._check_answer_set_existence(program)
            tensor_val = 1 if answer_set_exists else 0
            target_tensor = np.array(tensor_val)
            target_predictions = [
                [target_tensor]
            ]
        return target_predictions