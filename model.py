import time
import requests
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss


from llama2 import Llama
from clip import load_clip, test_image_encode
from dataset import download_pkl

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
      if int(int_part) >= 100000: int_part = '99999'
      if int(dec_part) >= 100000: dec_part = '99999'
      int_part = f"{int(int_part):05}"
      chunks = [self.markups[i*2], sign, int_part, '.', dec_part, self.markups[i*2+1]]
      result.extend(chunks)
    return result

  def encode(self, text):
    # Tokenizer.encode('0.2835957 0.10540066 0.6847595 -0.56894016 0.039462574 0.043872885 0.2')
    # print('tokenizer.encode', text)
    split_text = self.segment_string(text)
    return [self.token_to_id[token] for token in split_text]

  def decode(self, token_ids):
    return ' '.join(self.id_to_token[token_id] for token_id in token_ids)

# model

class ModelArgs:
  dim = 4096 # 4096
  depth = 8 # 32
  heads = 8 # 32
  vocab_size = -1  
  output_len = 7
  ffn_mult = 2
  norm_eps = 1e-5
  max_batch_size = 32 # 32
  max_seq_len = 1024 # 4096
  max_state_seq_len = 42 # 4096
  text_is_encoded = False
  image_is_encoded = False
  image_dim = 1024
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

class RobotTransformer(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.text_is_encoded = args.text_is_encoded
    self.image_is_encoded = args.image_is_encoded
    
    if not args.image_is_encoded:
      self.image_model, self.image_preproc = load_clip("ViT-L/14@336px")
      self.image_model.cuda().eval()
    self.image_mlp = nn.Sequential(
      nn.Linear(args.image_dim, args.dim),
      nn.ReLU(),
      nn.Linear(args.dim, args.dim)
    )  

    if not args.text_is_encoded:
      self.text_model = Llama()    
    self.text_mlp = nn.Sequential(
      nn.Linear(args.dim, args.ffn_mult * args.dim),
      nn.ReLU(),
      nn.Linear(args.ffn_mult * args.dim, args.dim)
    )

    self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
    self.pos_embeddings = nn.Embedding(args.max_state_seq_len, args.dim) 
    self.layers = torch.nn.ModuleList()
    for n in range(args.depth):
      self.layers.append(TransformerBlock(args))
    self.norm = LayerNorm(args.dim, eps=args.norm_eps)    
    self.reg_output = nn.Linear(args.dim, args.output_len)

  def forward(self, image, text, state):
    _bsz, seqlen = state.shape
    if not self.image_is_encoded:
      image = self.image_preproc(image).unsqueeze(0).to("cuda")
      image = self.image_model(image).to("cuda", dtype=torch.float32)
      image_clone = image.clone().detach() 
      image_clone.requires_grad_(True)
      image = image_clone
    image = self.image_mlp(image)

    if not self.text_is_encoded:
      text = self.text_model.encode_text(text).to("cuda", dtype=torch.float32)
      text_clone = text.clone().detach() 
      text_clone.requires_grad_(True)
      text = text_clone
    text = self.text_mlp(text)

    h = self.tok_embeddings(state)
    positions = torch.arange(0, seqlen, dtype=torch.long, device=h.device)
    h += self.pos_embeddings(positions)
    combined = torch.cat([h, image, text], dim=1)

    for layer in self.layers:
      h = layer(h)

    h = self.norm(h)
    output = self.reg_output(h.mean(dim=1)) 
    return output

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
  def __init__(self, n):
    self.data = []
    self.tokenizer = Tokenizer()
    self.load_data(n)

  def add_data(self, new_data):
    self.data.extend(new_data)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    step = self.data[idx]
    instr = step['text_encode'].squeeze(0)#.to("cuda", dtype=torch.float32)
    image = step['image_encode'].squeeze(0)#.to("cuda", dtype=torch.float32)
    state = torch.tensor(self.tokenizer.encode(step['src']))
    tgt = torch.tensor(step['tgt'])#.to("cuda", dtype=torch.float32)
    print(instr.shape)
    print(image.shape)
    print(state.shape)
    print(tgt.type, tgt.shape)
    return instr, image, state, tgt

  def load_data(self, count):
    for n in range(0, count):
      ep = download_pkl('bcz-encode', f'{n}.pkl')
      for step in ep:
        self.add_data([{
          'text_encode': step['text_encode'],
          'image_encode': step['image_encode'],
          'src': step['src'],
          'tgt': step['tgt'],
        }])

# data_loader = DataLoader(ds, batch_size=2, shuffle=True)
# ds = CustomDataset()
# print(len(ds))
# ds.load_data(3)
# print(len(ds))

class RT:
  def __init__(self):
    start_time = time.time()
    tokenizer = Tokenizer()
    model_args = ModelArgs()
    model_args.text_is_encoded = True
    model_args.image_is_encoded = True
    model_args.vocab_size = len(tokenizer.token_to_id)
    model = RobotTransformer(model_args)
    # checkpoint = torch.load("consolidated.00.pth", map_location="cuda")
    # model.load_state_dict(checkpoint, strict=False)
    self.model = model
    self.tokenizer = tokenizer   
  
  def train(self):
    batch_size = 16
    optimizer = Adam(self.model.parameters(), lr=1e-3)
    loss_fn = MSELoss()

    for n in range(0, 20):
      ep_loss = 0.0
      num_batches = 0

      # Create the dataset and data loader for the current episode
      print('creating dataset')
      dataset = CustomDataset(5)
      data_loader = DataLoader(dataset, batch_size=batch_size)#, shuffle=True)
      print('running training')
      for batch_instrs, batch_images, batch_states, batch_tgts in data_loader:
        continue
        instrs = batch_instrs.to("cuda", dtype=torch.float32)
        images = batch_images.to("cuda", dtype=torch.float32)
        states = torch.tensor(batch_states).to('cuda')
        tgts = torch.tensor(batch_tgts).to("cuda", dtype=torch.float32)

        model_output = self.model(images, instrs, states)

        loss = loss_fn(model_output, tgts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
        num_batches += 1

      average_loss = ep_loss / num_batches
      print(f'{n}\t{average_loss:.5f}')


rt = RT()
rt.train()
