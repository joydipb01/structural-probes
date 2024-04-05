import copy
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertOutput, BertLayer

class BertOutputWithoutFfn(BertOutput):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(hidden_states)

class BertLayerWithoutFfn(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.output2 = BertOutputWithoutFfn(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output2(intermediate_output, attention_output)
        return layer_output

class BertEncoderWithoutFfn(nn.Module):
    def __init__(self, config):
        super(BertEncoderWithoutFfn, self).__init__()
        layer = BertLayerWithoutFfn(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertWithoutFfn(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder2 = BertEncoderWithoutFfn(config)
    
    def forward(self, input_ids, token_type_ids, attention_mask = None, output_all_encoded_layers = True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder2(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output