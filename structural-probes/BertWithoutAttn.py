import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input_tensors):
        if len(input_tensors) != 1:
            return input_tensors[0]
        else:
            return input_tensors

class BertWithoutAttn(BertModel):
    def __init__(self, config):
        super().__init__(config)

        for layer in self.encoder.layer:
            layer.attention.self = Identity()
    
    def forward(self, input_ids, token_type_ids, attention_mask = None, output_all_encoded_layers = True):
        return super().forward(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)