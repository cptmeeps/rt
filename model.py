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

  def test(self):
    encoded = tokenizer.encode('+ 415 . 19001')
    print(f'encoded tokens\t\t[100000, 415, 100002, 19001]\n\t\t\t{encoded}')
    decoded = tokenizer.decode(encoded)
    print(decoded)  # Output should be the original string "+ 415 . 19001"
    print(f'decoded string\t\t+ 415 . 19001\n\t\t\t{decoded}')



