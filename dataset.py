import os
import pickle
import boto3
import numpy as np

# s3 utils

s3_client = boto3.client(
  's3',a
  region_name = os.getenv('aws_region_name'),
  aws_access_key_id = os.getenv('aws_access_key_id'),
  aws_secret_access_key = os.getenv('aws_secret_access_key'),
)  

def download_pkl():
  s3_client.download_file(bucket_name, file_name, file_name)
  with open(file_name, 'rb') as file:
    data = pickle.load(file)
  os.remove(file_name)
  return data

def get_obj_count(bucket_name, display=True):
  paginator = s3_client.get_paginator('list_objects_v2')
  page_iterator = paginator.paginate(Bucket=bucket_name)
  object_count = 0
  for page in page_iterator:
    object_count += len(page.get('Contents', []))
  if display: print(bucket_name, ' obj count: ', object_count)
  return object_count

def upload_pkl(dictionary, bucket_name, file_name):
  file_name = file_name + '.pkl'
  with open(file_name, 'wb') as file:
    pickle.dump(dictionary, file)
  with open(file_name, 'rb') as file:
    s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=file)
  os.remove(file_name)

# bc_z export
  
def extract_step(current, next, episode_id, step_count):
  current_obs = current['observation']
  next_obs = next['observation']
  step_data = {
    # metadata
    'id' : str(episode_id) + '_' + str(step_count),
    'is_first' : current['is_first'].numpy(),
    'is_last' : current['is_last'].numpy(),
    'is_terminal' : current['is_terminal'].numpy(),
    'episode_success' : current_obs['episode_success'].numpy(),
    'intervention' : current_obs['present/intervention'].numpy(),
    'autonomous' : current_obs['present/autonomous'].numpy(),
    'ttl_step_count' : current_obs['sequence_length'].numpy(),
    'current_step_count' : step_count,
    # source
    'image' : current_obs['image'].numpy(),
    'nl_instructions' : current_obs['natural_language_instruction'].numpy().decode('utf-8'),
    'nl_embedding_bcz' : current_obs['natural_language_embedding'].numpy(),
    'current_axis' : current_obs['present/axis_angle'].numpy(),
    'current_xyz' : current_obs['present/xyz'].numpy(),
    'current_gripper' : current_obs['present/sensed_close'].numpy(),
    # target
    'next_axis' : next_obs['present/axis_angle'].numpy(),
    'next_xyz' : next_obs['present/xyz'].numpy(),
    'next_gripper' : next_obs['present/sensed_close'].numpy(),
    'future_axis_res' : current['action']['future/axis_angle_residual'].numpy(),
    'future_xyz_res' : current['action']['future/xyz_residual'].numpy(),
    'future_gripper' : current['action']['future/target_close'].numpy(),
  }
  return step_data

def extract_episode(episode, episode_id):
  steps = list(episode['steps'])
  steps_len = len(steps)
  output = []
  for i in range(0, steps_len - 1):
    step_data = extract_step(steps[i], steps[i+1], episode_id, i)
    output.append(step_data)
  last_step = steps[steps_len - 1]
  step_data = extract_step(last_step, last_step, episode_id, steps_len)
  output.append(step_data)
  return output
    
def export_batch():
  start_index = get_obj_count('bcz-pkl')
  batch_size = 1000 
  split_str = f'train[{start_index}:{start_index+batch_size}]'
  print(split_str)
  url = 'gs://gresearch/robotics/bc_z/0.1.0'
  ds_builder = tfds.builder_from_directory(builder_dir=url)
  ds = ds_builder.as_dataset(split=split_str)
  episode_id = start_index
  for episode in ds:
    print('episode: ', episode_id)
    episode_data = extract_episode(episode, episode_id)
    upload_pkl(episode_data, 'bcz-pkl', str(episode_id))
    episode_id += 1

# bc_z encode

ep = download_pkl('bcz-pkl', '1.pkl')
print(len(ep))








