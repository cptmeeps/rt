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
    self.token_to_id = {f"{num:05}": num for num in range(0, 100000)}
    self.id_to_token = {num: f"{num:05}" for num in range(0, 100000)} 
    self.symbols = ['+', '-', '.',]
    self.markups = [ 
      'pos_x_start', 'pos_x_end',
      'pos_y_start', 'pos_y_end',
      'pos_z_start', 'pos_z_end',
      'angle_x_start', 'angle_x_end',
      'angle_y_start', 'angle_y_end',
      'angle_z_start', 'angle_z_end',
      'grip_open', 'grip_close'
    ]
    self.special_tokens = self.symbols + self.markups

    start_id = 100000
    for token in self.special_tokens:
      self.token_to_id[token] = start_id
      self.id_to_token[start_id] = token
      start_id += 1

  def segment_string(self, float_string):
    float_list = float_string.split()
    result = []
    for i, num in enumerate(float_list):
      sign = '+' if float(num) >= 0 else '-'
      int_part, dec_part = num.lstrip('-+').split('.')
      dec_part = f"{int(round(float('0.' + dec_part), 5) * 100000):05}"
      int_part = f"{int(int_part):05}"
      chunks = [self.markups[i*2], sign, int_part, '.', dec_part, self.markups[i*2+1]]
      result.extend(chunks)
    return result

  def encode(self, text):
    # Tokenizer.encode('0.2835957 0.10540066 0.6847595 -0.56894016 0.039462574 0.043872885 0.2')
    split_text = self.segment_string(text)
    return [self.token_to_id[token] for token in split_text]

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
