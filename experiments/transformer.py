import torch
import torch.nn as nn
import math

from utils import get_device

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        position = torch.arange(max_len).unsqueeze(1)
        pos_emb = torch.zeros(max_len, 1, d_model)

        pos_emb[:, 0, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pos_emb', pos_emb)
    
    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x + self.pos_emb[:x.size(0)]
        x = self.dropout(x).permute(1, 0, 2)
        return x

class LatentConceptTransformer(nn.Module):
    def __init__(self,
                metadata,
                output_size,
                latent_embedding_sizes,
                symbolic_embedding_size=128,
                hidden_size=128,
                n_heads=4,
                dropout_value=0.1,
                transformer_ffn_dim=512,
                num_transformer_encoder_layers=1,
                transformer_activation='gelu'
    ):
        super().__init__()

        self.input_size = self.__get_raw_inputs_total_size(metadata, latent_embedding_sizes)
        self.input_size += self.__setup_symbolic_inputs(metadata, symbolic_embedding_size)
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.n_heads = n_heads
        self.dropout_value = dropout_value
        self.transformer_ffn_dim = transformer_ffn_dim
        self.num_transformer_encoder_layers = num_transformer_encoder_layers
        self.transformer_activation = transformer_activation
        self.sequence_length = len(metadata.raw_inputs.values()) + len(metadata.symbolic_inputs.values())
        self.input_size = symbolic_embedding_size
        self.device = get_device()

        self.__setup_reasoning_layers()


    def __get_raw_inputs_total_size(self, metadata, latent_embedding_sizes):
        return sum([latent_embedding_sizes[raw_input_data.concept_name]
                    for raw_input_data in metadata.raw_inputs.values()])

    def __setup_symbolic_inputs(self, metadata, symbolic_embedding_size):
        self.__symbolic_embeddings = nn.ModuleDict()
        self.__symbolic_name_to_embedding_layer = dict()
        symbolic_input_size = 0
        for symbolic_input_name, symbolic_input_data in metadata.symbolic_inputs.items():
            if symbolic_input_data.concept_name not in self.__symbolic_name_to_embedding_layer:
                num_embeddings = len(metadata.symbolic_concepts[symbolic_input_data.concept_name].values)
                embedding_layer = nn.Embedding(num_embeddings=num_embeddings,
                                                     embedding_dim=symbolic_embedding_size)
                self.__symbolic_embeddings[symbolic_input_data.concept_name] = embedding_layer
            self.__symbolic_name_to_embedding_layer[symbolic_input_name] = self.__symbolic_embeddings[
                symbolic_input_data.concept_name]
            symbolic_input_size += symbolic_embedding_size
        return symbolic_input_size

    def __setup_reasoning_layers(self):
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.transformer_ffn_dim,
            dropout=self.dropout_value,
            activation=self.transformer_activation,
            batch_first=True,
            device=self.device
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer,
            num_layers=self.num_transformer_encoder_layers
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, self.sequence_length, self.hidden_size))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(self.dropout_value)

        self.layer_norm = nn.LayerNorm(self.hidden_size)

    
    def __get_symbolic_embedings(self, symbolic_inputs):
        return [
            self.__symbolic_name_to_embedding_layer[symbolic_name](symbolic_input)
            for symbolic_input, symbolic_name in zip(symbolic_inputs, self.__symbolic_name_to_embedding_layer.keys())
        ]

    def __get_transformer_input(self, raw_input_embeddings, symbolic_inputs):
        symbolic_embeddings = self.__get_symbolic_embedings(symbolic_inputs)
        return torch.stack(raw_input_embeddings + symbolic_embeddings, dim=1)
        
  
    def forward(self, raw_input_embeddings, symbolic_inputs):
        x = self.__get_transformer_input(raw_input_embeddings, symbolic_inputs)

        positional_embeddings = self.pos_emb[:, :, :]
        x_positional = self.dropout(x + positional_embeddings)

        x_transformer = self.transformer(x_positional)

        x_pooled, _ = torch.max(x_transformer, dim=1)

        x_layer_norm = self.layer_norm(x_pooled)
        
        return x_layer_norm
