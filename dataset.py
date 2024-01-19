import tqdm
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class RoboticsDataset(Dataset):
  def __init__(self, tf_dataset):
    self.tf_dataset = list(tf_dataset)

  def __len__(self):
    return len(self.tf_dataset)

  def __getitem__(self, idx):
    episode = self.tf_dataset[idx]
    output = self.process_episode(episode)
    return output

def process_bcz_episode(episode):
  steps = list(episode['steps'])
  output = []
  tgt = []
  for i in range(0, len(steps) - 1):
    current_step = steps[i]
    next_step = steps[i + 1]
    src = {
      'image' : current_step['observation']['image'],
      'instruction' : current_step['observation']['natural_language_instruction'],
      'axis_angle' : current_step['observation']['present/axis_angle'],
      'xyz' : current_step['observation']['present/xyz'],
      'sensed_close' : current_step['observation']['present/sensed_close'],
    }
    tgt = {
      'axis_angle' : next_step['observation']['present/axis_angle'],
      'xyz' : next_step['observation']['present/xyz'],
      'sensed_close' : next_step['observation']['present/sensed_close'],
    }
    output.append([src, tgt])

def sample_bcz():
    b = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bc_z/0.1.0')
    tf_dataset = b.as_dataset(split='train[:30]')#.shuffle(10) 

def download_bcz():
  print(f"downloading bcz")
  _ = tfds.load('bc_z', data_dir='~/bcz_dataset')

  
    

download_bcz()
