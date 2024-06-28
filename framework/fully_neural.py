import torch
import torch.nn as nn

class FullyNeuralModel(torch.nn.Module):
    def __init__(self, metadata, perception_networks, reasoning_network):
        super().__init__()
        self._metadata = metadata
        self.perception_networks = torch.nn.ModuleDict(
            {
                concept_name: network
                for concept_name, network in perception_networks.items()
            }
        )
        self.reasoning_network = reasoning_network
        self._setup_class_heads()
        self.apply(self._init_weights)

    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def _setup_class_heads(self):
        self.class_heads = torch.nn.ModuleDict({})
        self.target_to_class_head = list()
        for target_name, target_data in self._metadata.targets.items():
            if target_data.target_concept_name not in self.class_heads:
                num_classes = len(self._metadata.target_concepts[target_data.target_concept_name].values)
                if target_data.multi_valued:
                    self.class_heads[target_data.target_concept_name] = torch.nn.Sequential(
                        torch.nn.Linear(in_features=self.reasoning_network.output_size, out_features=num_classes),
                        torch.nn.Sigmoid()
                    )
                else:
                    self.class_heads[target_data.target_concept_name] = torch.nn.Sequential(
                        torch.nn.Linear(in_features=self.reasoning_network.output_size, out_features=num_classes),
                        torch.nn.Sigmoid()
                    )
            self.target_to_class_head.append(self.class_heads[target_data.target_concept_name])
   
            
    def forward(self, raw_inputs, symbolic_inputs, return_embeddings=False):
        raw_input_embeddings = list()
        for raw_input, raw_input_data in zip(raw_inputs, self._metadata.raw_inputs.values()):
            raw_input_embeddings.append(self.perception_networks[raw_input_data.concept_name](raw_input))
        if return_embeddings:
            return raw_input_embeddings
        output_embedding = self.reasoning_network(raw_input_embeddings, symbolic_inputs)
        outputs = list()
        for class_head in self.target_to_class_head:
            outputs.append(class_head(output_embedding))
        return outputs


