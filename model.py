import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import math
from torch import nn, optim


def train_epoch(model, data_loader, loss_fn, optimizer, device):
  model.train()
  total_loss = 0
  for batch in data_loader:
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  average_loss = total_loss / len(data_loader)
  return average_loss

def save_checkpoint(model, optimizer, epoch, filename):
  checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch
  }
  torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, checkpoint_path):
  if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming training from epoch {epoch+1}")
    return epoch + 1  # Return the next epoch
  else:
    print("No checkpoint found. Starting training from scratch.")
    return 0  # Start from the first epoch

def run_training(model, data_loader, epochs=10, checkpoint_path=None):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  loss_fn = nn.MSELoss() 
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  start_epoch = 0
  if checkpoint_path:
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
  for epoch in range(start_epoch, epochs):
    avg_loss = train_epoch(model, data_loader, loss_fn, optimizer, device)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    if checkpoint_path:
      save_checkpoint(model, optimizer, epoch, f"{checkpoint_path}_epoch_{epoch+1}.pt")

class Tokenizer:
  def __init__(self):
    self.token_to_id = {str(num): num for num in range(1, 100000)}
    self.id_to_token = {num: str(num) for num in range(1, 100000)}
    
    special_tokens = [
      '+', '-','.',
      'a_x_s', 'a_x_e',
      'a_t_s', 'a_y_e',
      'a_z_s', 'a_z_e',
      'c_x_s', 'c_x_e',
      'c_y_s', 'c_y_e',
      'c_z_s', 'c_z_e',
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

class ModelArgs:
  dim = 4096
  n_layers = 32
  n_heads = 32
  vocab_size = -1  
  output_len = 6
  ffn_mult = None
  norm_eps = 1e-5
  max_batch_size = 1 # 32
  max_seq_len = 1024 # 4096

class TransformerBlock(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.attention = nn.MultiheadAttention(embed_dim=args.dim, num_heads=args.n_heads, batch_first=True)
    self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
    self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
    self.feed_forward = nn.Sequential(
      nn.Linear(args.dim, args.ffn_mult * args.dim),
      nn.ReLU(),
      nn.Dropout(args.dropout),
      nn.Linear(args.ffn_mult * args.dim, args.dim)
    )

  def forward(self, x):
    x_norm = self.attention_norm(x)
    attn_output, _ = self.attention(x_norm, x_norm, x_norm)
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
    for n in range(args.n_layers):
      self.layers.append(TransformerBlock(args))
    self.norm = LayerNorm(args.dim, eps=args.norm_eps)    
    self.reg_output = nn.Linear(args.dim, args.output_len)

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