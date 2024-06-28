import torch


class MLPReasoning(torch.nn.Module):
    def __init__(self, config, metadata, latent_embedding_sizes):
        super().__init__()
        self._config = config
        self.mlp_input_size = self._get_raw_inputs_total_size(metadata, latent_embedding_sizes)
        self.mlp_input_size += self._setup_symbolic_inputs(metadata, config["model"]["symbolic_embedding_size"])
        self._setup_reasoning_layers(self.mlp_input_size, config["model"]["reasoning_hidden_size"], config["model"]["reasoning_output_size"])
        self.output_size = config["model"]["reasoning_output_size"]

    def _get_raw_inputs_total_size(self, metadata, latent_embedding_sizes):
        return sum([latent_embedding_sizes[raw_input_data.concept_name]
                    for raw_input_data in metadata.raw_inputs.values()])

    def _setup_symbolic_inputs(self, metadata, symbolic_embedding_size):
        self._symbolic_embeddings = torch.nn.ModuleDict()
        self._symbolic_name_to_embedding_layer = dict()
        symbolic_input_size = 0
        for symbolic_input_name, symbolic_input_data in metadata.symbolic_inputs.items():
            if symbolic_input_data.concept_name not in self._symbolic_name_to_embedding_layer:
                num_embeddings = len(metadata.symbolic_concepts[symbolic_input_data.concept_name].values)
                embedding_layer = torch.nn.Embedding(num_embeddings=num_embeddings,
                                                     embedding_dim=symbolic_embedding_size)
                self._symbolic_embeddings[symbolic_input_data.concept_name] = embedding_layer
            self._symbolic_name_to_embedding_layer[symbolic_input_name] = self._symbolic_embeddings[
                symbolic_input_data.concept_name]
            symbolic_input_size += symbolic_embedding_size
        return symbolic_input_size

    def _setup_reasoning_layers(self, mlp_input_size, hidden_size, output_size):
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=mlp_input_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=output_size),
            torch.nn.ReLU()
        )

    def _get_symbolic_embedings(self, symbolic_inputs):
        return [
            self._symbolic_name_to_embedding_layer[symbolic_name](symbolic_input)
            for symbolic_input, symbolic_name in zip(symbolic_inputs, self._symbolic_name_to_embedding_layer.keys())
        ]

    def _get_mlp_input(self, raw_input_embeddings, symbolic_inputs):
        symbolic_embeddings = self._get_symbolic_embedings(symbolic_inputs)
        return torch.cat(raw_input_embeddings + symbolic_embeddings, dim=-1)


    def forward(self, raw_input_embeddings, symbolic_inputs):
        mlp_input = self._get_mlp_input(raw_input_embeddings, symbolic_inputs)
        out = self.layers(mlp_input)
        return out


class ResBlock(torch.nn.Module):
    def __init__(self, features, batch_norm = True):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=features, out_features=features),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(num_features=features) if batch_norm else torch.nn.Identity()
        )

    def forward(self, x):
        r = x.clone()
        x = self.layers(x)
        return x+r

class ResBlockReasoning(MLPReasoning):
    def __init__(self, config, metadata, latent_embedding_sizes):
        super().__init__(config, metadata, latent_embedding_sizes)
        self.__setup_reasoning_layers(self.mlp_input_size, config["model"]["reasoning_hidden_size"], config["model"]["reasoning_output_size"])

    def __setup_reasoning_layers(self, mlp_input_size, hidden_size, output_size):
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=mlp_input_size, out_features=hidden_size),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(num_features=hidden_size),
            *[ResBlock(features=hidden_size) for _ in range(self._config["model"]["num_res_blocks"])],
            torch.nn.Linear(in_features=hidden_size, out_features=output_size),
        )



