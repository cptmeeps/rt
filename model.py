import time
import requests
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.nn import LayerNorm
import torch.nn.functional as F
from llama2 import Llama
from clip import load_clip, test_image_encode

# tokenizer

class Tokenizer:
  def __init__(self):
    self.token_to_id = {str(num): num for num in range(1, 100000)}
    self.id_to_token = {num: str(num) for num in range(1, 100000)}
    
    special_tokens = [
      '+', '-', '.',
      'a_x_s', 'a_y_s', 'a_z_s', 'a_x_e', 'a_y_e', 'a_z_e',
      'c_x_s', 'c_y_s', 'c_z_s', 'c_x_e', 'c_y_e', 'c_z_e',
      'grip_open', 'grip_close'
    ]

    start_id = 100000
    for token in special_tokens:
      self.token_to_id[token] = start_id
      self.id_to_token[start_id] = token
      start_id += 1

  def encode(self, text):
    return [self.token_to_id[token] for token in text.split()]

  def decode(self, token_ids):
    return ' '.join(self.id_to_token[token_id] for token_id in token_ids)

# model

class ModelArgs:
  dim = 4096
  depth = 32
  heads = 32
  vocab_size = -1  
  output_len = 6
  ffn_mult = 2
  norm_eps = 1e-5
  max_batch_size = 1 # 32
  max_seq_len = 1024 # 4096
  dropout = 0.1

class TransformerBlock(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.attn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
    self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
    self.attn = nn.MultiheadAttention(embed_dim=args.dim, num_heads=args.heads, batch_first=True)
    self.feed_forward = nn.Sequential(
      nn.Linear(args.dim, args.ffn_mult * args.dim),
      nn.ReLU(),
      nn.Dropout(args.dropout),
      nn.Linear(args.ffn_mult * args.dim, args.dim)
    )

  def forward(self, x):
    x_norm = self.attn_norm(x)
    attn_output, _ = self.attn(x_norm, x_norm, x_norm)
    h = x + attn_output
    h_norm = self.ffn_norm(h)
    out = h + self.feed_forward(h_norm)
    return out

class Transformer(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
    self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim) 
    self.layers = torch.nn.ModuleList()
    for n in range(args.depth):
      self.layers.append(TransformerBlock(args))
    self.norm = LayerNorm(args.dim, eps=args.norm_eps)    
    self.reg_output = nn.Linear(args.dim, args.output_len)
    
    self.image_encode, self.image_preprocess = load_clip("ViT-L/14@336px")
    self.image_encode.cuda().eval()
    self.text_encode = Llama()
    

  def forward(self, tokens):
    _bsz, seqlen = tokens.shape

    h = self.tok_embeddings(tokens)
    positions = torch.arange(0, seqlen, dtype=torch.long, device=h.device)
    h += self.pos_embeddings(positions)
    
    for layer in self.layers:
      h = layer(h)
    h = self.norm(h)
    output = self.reg_output(h.mean(dim=1)) 
    return output

class RT:
  def __init__(self):
    start_time = time.time()
    tokenizer = Tokenizer()
    model_args = ModelArgs()
    model_args.vocab_size = 100000#tokenizer.n_words
    # checkpoint = torch.load("consolidated.00.pth", map_location="cuda")
    model = Transformer(model_args)
    # model.load_state_dict(checkpoint, strict=False)
    self.model = model
    self.tokenizer = tokenizer   
