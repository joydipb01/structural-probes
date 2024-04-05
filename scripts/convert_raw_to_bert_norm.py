'''
Takes raw text and saves BERT-cased features for that text to disk

Adapted from the BERT readme (and using the corresponding package) at

https://github.com/huggingface/pytorch-pretrained-BERT

###
John Hewitt, johnhew@stanford.edu
Feb 2019

'''
import torch
import torch.nn.functional as F
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
from transformers import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
from argparse import ArgumentParser
import h5py
import numpy as np

argp = ArgumentParser()
argp.add_argument('input_path')
argp.add_argument('output_path')
argp.add_argument('bert_model', help='base or large')
args = argp.parse_args()

def pad_to_feature_count(tensor, feature_count):
  pad_width = (0, feature_count - tensor.size(2))
  padded_tensor = F.pad(tensor, pad_width, mode='constant', value=0)
  return padded_tensor

# Load pre-trained model tokenizer (vocabulary)
# Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
if args.bert_model == 'base':
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
  model = BertModel.from_pretrained('bert-base-cased')
  LAYER_COUNT = 12
  FEATURE_COUNT = 768
elif args.bert_model == 'large':
  tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
  model = BertModel.from_pretrained('bert-large-cased')
  LAYER_COUNT = 24
  FEATURE_COUNT = 1024
else:
  raise ValueError("BERT model must be base or large")

model.eval()

with h5py.File(args.output_path, 'w') as fout:
  for index, line in enumerate(open(args.input_path)):
    line = line.strip() # Remove trailing characters
    line = '[CLS] ' + line + ' [SEP]'
    tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [1 for x in tokenized_text]
  
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
  
    with torch.no_grad():
        encoded_layers = model(tokens_tensor, segments_tensors, output_norms = True)
        #print(len(encoded_layers[2][0][1]))
    dset = fout.create_dataset(str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
    # encoded_layers[2][0] is a tuple of 7 elements (Kobayashi et al., 2021)
    # 0: head-level Attn-N from Kobayashi et al., 2020
    # 1: layer-level Attn-N from Kobayashi et al., 2020
    # 2: AttnRes-N
    # 3: AttnResLn-N
    # 4: Mixing ratios for Attn-N
    # 5: Mixing ratios for AttnRes-N
    # 6: Mixing ratios for AttnResLn-N
    dset[:,:] = np.vstack([np.array(x) for x in pad_to_feature_count(encoded_layers[2][0][2], FEATURE_COUNT)])
  

