import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import math
from torch import nn, optim
import numpy as np
from torch.utils.data import Dataset

class RoboticsDataset(Dataset):
  def __init__(self, tf_dataset):
    self.tf_dataset = list(tf_dataset)  # Convert to list for easier indexing
    self.batch_size = 8

  def __len__(self):
    return len(self.tf_dataset)

  def __getitem__(self, idx):
    episode = self.tf_dataset[idx]
    steps = list(episode['steps'])  # Convert to list for easier indexing

    inputs = []
    labels = []

    for i in range(0, len(steps) - 1, self.batch_size):
      batch_inputs = []
      batch_labels = []

      for j in range(i, min(i + self.batch_size, len(steps) - 1)):
        current_step = steps[j]
        next_step = steps[j + 1]

        image = current_step['observation']['image'].numpy()
        image = image.astype(np.float32)
        instruction = current_step['observation']['natural_language_instruction'].numpy()
        axis_angle = current_step['observation']['present/axis_angle'].numpy()
        xyz = current_step['observation']['present/xyz'].numpy()
        sensed_close = current_step['observation']['present/sensed_close'].numpy()
        current_state = np.concatenate((axis_angle, xyz, sensed_close), axis=0)
        input_features = {
          'image' : image,
          'instruction' : instruction,
          'state' : current_state
        }

        label_axis_angle = next_step['observation']['present/axis_angle'].numpy()
        label_xyz = next_step['observation']['present/xyz'].numpy()
        label_sensed_close = next_step['observation']['present/sensed_close'].numpy()
        label_state = np.concatenate((label_axis_angle, label_xyz, label_sensed_close), axis=0)

        batch_inputs.append(input_features)
        batch_labels.append(label_state)

      inputs.append(batch_inputs)
      labels.append(batch_labels)

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return inputs_tensor, labels_tensor

tf_dataset = ds
torch_dataset = RoboticsDataset(tf_dataset)
data_loader = DataLoader(torch_dataset, batch_size=1, shuffle=True)

# train

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

def run_training(model, data_loader, epochs=10, checkpoint_path=None):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  loss_fn = nn.MSELoss() 
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  start_epoch = 0

  if checkpoint_path and os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch+1}")
  else:
    print("No checkpoint found. Starting training from scratch.")

  for epoch in range(start_epoch, epochs):
    avg_loss = train_epoch(model, data_loader, loss_fn, optimizer, device)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    if checkpoint_path:
      checkpoint_filename = f"{checkpoint_path}_epoch_{epoch+1}.pt"
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
      }, checkpoint_filename)

# model

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

class ModelArgs:
  dim = 4096
  depth = 32
  heads = 32
  vocab_size = -1  
  output_len = 6
  ffn_mult = None
  norm_eps = 1e-5
  max_batch_size = 1 # 32
  max_seq_len = 1024 # 4096

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