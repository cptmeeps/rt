import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
