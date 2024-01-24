import os
import hashlib
import urllib
import requests
import warnings
import time
from collections import OrderedDict
from pkg_resources import packaging
import time
import json
import math
from typing import Optional, Tuple, List, TypedDict, Any, Union
from dataclasses import dataclass
from sentencepiece import SentencePieceProcessor

from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# helpers

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_tensor_type('torch.cuda.HalfTensor')

# clip

_MODELS = {
  "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
  "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
  "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
  "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

# clip model architecture

class LayerNorm(nn.LayerNorm):
  def forward(self, x: torch.Tensor):
    orig_type = x.dtype
    ret = super().forward(x.type(torch.float32))
    return ret.type(orig_type)

class QuickGELU(nn.Module):
  def forward(self, x: torch.Tensor):
    return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
  def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
    super().__init__()
    self.attn = nn.MultiheadAttention(d_model, n_head)
    self.ln_1 = LayerNorm(d_model)
    self.mlp = nn.Sequential(OrderedDict([
      ("c_fc", nn.Linear(d_model, d_model * 4)),
      ("gelu", QuickGELU()),
      ("c_proj", nn.Linear(d_model * 4, d_model))
    ]))
    self.ln_2 = LayerNorm(d_model)
    self.attn_mask = attn_mask

  def attention(self, x: torch.Tensor):
    self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

  def forward(self, x: torch.Tensor):
    x = x + self.attention(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class Transformer(nn.Module):
  def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
    super().__init__()
    self.width = width
    self.layers = layers
    self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

  def forward(self, x: torch.Tensor):
    return self.resblocks(x)

class VisionTransformer(nn.Module):
  def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
    super().__init__()
    self.input_resolution = input_resolution
    self.output_dim = output_dim
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

    scale = width ** -0.5
    self.class_embedding = nn.Parameter(scale * torch.randn(width))
    self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
    self.ln_pre = LayerNorm(width)

    self.transformer = Transformer(width, layers, heads)

    self.ln_post = LayerNorm(width)
    self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

  def forward(self, x: torch.Tensor):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = self.ln_post(x[:, 0, :])

    if self.proj is not None:
        x = x @ self.proj

    return x

class CLIP(nn.Module):
  def __init__(self,
          embed_dim: int,
          # vision
          image_resolution: int,
          vision_layers: Union[Tuple[int, int, int, int], int],
          vision_width: int,
          vision_patch_size: int,
          # text
          context_length: int,
          vocab_size: int,
          transformer_width: int,
          transformer_heads: int,
          transformer_layers: int
          ):
    super().__init__()
    self.context_length = context_length

    vision_heads = vision_width // 64
    self.visual = VisionTransformer(
      input_resolution=image_resolution,
      patch_size=vision_patch_size,
      width=vision_width,
      layers=vision_layers,
      heads=vision_heads,
      output_dim=embed_dim
    )

    self.transformer = Transformer(
      width=transformer_width,
      layers=transformer_layers,
      heads=transformer_heads,
      attn_mask=self.build_attention_mask()
    )

    self.vocab_size = vocab_size
    self.token_embedding = nn.Embedding(vocab_size, transformer_width)
    self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
    self.ln_final = LayerNorm(transformer_width)

    self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    self.initialize_parameters()

  def initialize_parameters(self):
    nn.init.normal_(self.token_embedding.weight, std=0.02)
    nn.init.normal_(self.positional_embedding, std=0.01)

    proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
    attn_std = self.transformer.width ** -0.5
    fc_std = (2 * self.transformer.width) ** -0.5
    for block in self.transformer.resblocks:
      nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
      nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
      nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
      nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    if self.text_projection is not None:
      nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

  def build_attention_mask(self):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(self.context_length, self.context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask

  @property
  def dtype(self):
    return self.visual.conv1.weight.dtype

  def encode_image(self, image):
    return self.visual(image.type(self.dtype))
  
  def forward(self, image):
    image_features = self.encode_image(image)
    return image_features

def convert_weights(model: nn.Module):
  # Convert applicable model parameters to fp16

  def _convert_weights_to_fp16(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
      l.weight.data = l.weight.data.half()
      if l.bias is not None:
        l.bias.data = l.bias.data.half()

    if isinstance(l, nn.MultiheadAttention):
      for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
        tensor = getattr(l, attr)
        if tensor is not None:
            tensor.data = tensor.data.half()

    for name in ["text_projection", "proj"]:
      if hasattr(l, name):
        attr = getattr(l, name)
        if attr is not None:
           attr.data = attr.data.half()

  model.apply(_convert_weights_to_fp16)

def build_model(state_dict: dict):
  start_time = time.time()
  vision_width = state_dict["visual.conv1.weight"].shape[0]
  vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
  vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
  grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
  image_resolution = vision_patch_size * grid_size

  embed_dim = state_dict["text_projection"].shape[1]
  context_length = state_dict["positional_embedding"].shape[0]
  vocab_size = state_dict["token_embedding.weight"].shape[0]
  transformer_width = state_dict["ln_final.weight"].shape[0]
  transformer_heads = transformer_width // 64
  transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

  model = CLIP(
    embed_dim,
    image_resolution, vision_layers, vision_width, vision_patch_size,
    context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
  )

  for key in ["input_resolution", "context_length", "vocab_size"]:
    if key in state_dict:
      del state_dict[key]
  convert_weights(model)
  model.load_state_dict(state_dict, strict=False)
  print(f"clip loaded:\t{time.time() - start_time:.2f} seconds")
  print(f" mem alloc: {round(torch.cuda.memory_allocated(0) / (1024 ** 3))} GB")
  return model.eval()

def _download(url: str, root: str = '.'):
  os.makedirs(root, exist_ok=True)
  filename = os.path.basename(url)
  expected_sha256 = url.split("/")[-2]
  download_target = os.path.join(root, filename)

  if os.path.exists(download_target) and not os.path.isfile(download_target):
    raise RuntimeError(f"{download_target} exists and is not a regular file")

  if os.path.isfile(download_target):
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
      return download_target
    else:
      warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

  with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
    with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
      while True:
        buffer = source.read(8192)
        if not buffer:
          break

        output.write(buffer)
        loop.update(len(buffer))

  if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
    raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

  return download_target

def _convert_image_to_rgb(image):
  return image.convert("RGB")

def _transform(n_px):
  return Compose([
    Resize(n_px, interpolation=BICUBIC),
    CenterCrop(n_px),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
  ])

def load_clip(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
  if name in _MODELS:
    model_path = _download(_MODELS[name])
  elif os.path.isfile(name):
    model_path = name
  else:
    raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

  with open(model_path, 'rb') as opened_file:
    try:
      # loading JIT archive
      model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
      state_dict = None
    except RuntimeError:
      # loading saved state dict
      if jit:
        # warnings.warn(f"\tFile {model_path} is not a JIT archive. Loading as a state dict instead")
        jit = False
      state_dict = torch.load(opened_file, map_location="cpu")

  if not jit:
    model = build_model(state_dict or model.state_dict()).to(device)
    if str(device) == "cpu":
      model.float()
    return model, _transform(model.visual.input_resolution)

def test_image_encode():
  model, preprocess = load_clip("ViT-L/14@336px")
  model.cuda().eval()
  start_time = time.time()
  url = "http://images.cocodataset.org/val2017/000000039769.jpg"
  image = Image.open(requests.get(url, stream=True).raw)
  image = preprocess(image).unsqueeze(0).to("cuda")
  image_encoding = model(image)
  print("image_encoding.shape: ", image_encoding.shape)
  print(f"CLIP.encode_image\n\tencode in {time.time() - start_time:.2f} seconds")
  #print("image_encoding:\n ", image_encoding)


# llama 

class Tokenizer:
  def __init__(self, model_path: str):
    assert os.path.isfile(model_path), model_path
    self.sp_model = SentencePieceProcessor(model_file=model_path)
    self.n_words: int = self.sp_model.vocab_size()
    self.bos_id: int = self.sp_model.bos_id()
    self.eos_id: int = self.sp_model.eos_id()
    self.pad_id: int = self.sp_model.pad_id()
    assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

  def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
    assert type(s) is str
    t = self.sp_model.encode(s)
    if bos:
      t = [self.bos_id] + t
    if eos:
      t = t + [self.eos_id]
    return t

  def decode(self, t: List[int]) -> str:
    return self.sp_model.decode(t)

class MMP(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.w1 = nn.Linear(args.dim, args.dim, bias=True)
    self.gelu = nn.GELU()
    self.w2 = nn.Linear(args.dim, args.dim, bias=True)

  def forward(self, image_features):
    h = self.w1(image_features)
    h = self.gelu(h)
    h = self.w2(h)
    return hidden_states

@dataclass
class ModelArgs:
  dim = 4096
  h_dim = 4 * 4096
  n_layers = 32
  n_heads = 32
  # n_kv_heads = None
  vocab_size = -1  # defined later by tokenizer
  multiple_of = 256  # make SwiGLU hidden layer size multiple of large power of 2
  ffn_dim_multiplier = None
  norm_eps = 1e-5
  max_batch_size = 1 # 32
  max_seq_len = 4096

class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

def precompute_freqs_cis(dim, end, theta=10000.0):
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
  ndim = x.ndim
  assert 0 <= 1 < ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.n_heads = args.n_heads
    self.head_dim = args.dim // args.n_heads
    self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
 
  def forward(self, x, freqs_cis, mask=None):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

    # Apply rotary embeddings directly in the forward method
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # ndim = xq.ndim
    # shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq.shape)]
    # freqs_cis = freqs_cis.view(*shape)
    # xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # xq = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
    # xk = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
    xk = xk.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
    xv = xv.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
    scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None:
      scores = scores + mask  # (bs, n_heads, seqlen, seqlen)
    scores = F.softmax(scores.float(), dim=-1).type_as(scores)
    output = torch.matmul(scores, xv)  # (bs, n_heads, seqlen, head_dim)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)

class FeedForward(nn.Module):
  def __init__(self, dim, h_dim):
    super().__init__()
    self.w1 = nn.Linear(dim, h_dim, bias=False)
    self.w2 = nn.Linear(h_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, h_dim, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.dim = args.dim
    self.attn = Attention(args)
    self.ff = FeedForward(args.dim, args.h_dim)
    self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

  def forward(self, x, start_pos, freqs_cis, mask,):
    h = x + self.attn.forward(self.attn_norm(x), freqs_cis, mask)
    out = h + self.ff.forward(self.ffn_norm(h))
    return out

class Transformer(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.vocab_size = args.vocab_size
    self.n_layers = args.n_layers

    self.tok_emb = nn.Embedding(args.vocab_size, args.dim)

    self.layers = torch.nn.ModuleList()
    for layer_id in range(args.n_layers):
      self.layers.append(TransformerBlock(args))

    self.norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.output = nn.Linear(
      args.dim, args.vocab_size, bias=False
    )

    self.freqs_cis = precompute_freqs_cis(
      self.args.dim // self.args.n_heads, self.args.max_seq_len * 2
    )


  def forward(self, tokens: torch.Tensor, start_pos: int):
    _bsz, seqlen = tokens.shape
    h = self.tok_emb(tokens)
    self.freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

    mask = None
    if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)

    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis, mask)

    h = self.norm(h)
    output = self.output(h).float()
    return output

class Llama:
  def __init__(self):
    start_time = time.time()
    tokenizer = Tokenizer(model_path="tokenizer.model")
    model_args = ModelArgs()
    model_args.vocab_size = tokenizer.n_words
    checkpoint = torch.load("consolidated.00.pth", map_location="cuda")
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    print(f"llama loaded:\t{time.time() - start_time:.2f} seconds")
    print(f" mem alloc: {round(torch.cuda.memory_allocated(0) / (1024 ** 3))} GB")
    self.model = model
    self.tokenizer = tokenizer

  @torch.inference_mode()
  def encode_text(self, prompts):
    prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    bsz = len(prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    pad_id = self.tokenizer.pad_id
    tokens = torch.full((bsz, max_prompt_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
      tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    encodings = self.model.forward(tokens, 0)
    return encodings

  def test_encode(self):
    start_time = time.time()
    prompt = ["Tell me 3 facts about the color red"]
    encodings = self.encode_text(prompt)
    print(f"expected values:")
    print(f"\t1   [-0.0799,  1.6094,  3.1934,  ...,  1.3057, -0.8794,  1.9229]")
    print(f"\t2   [ 3.8477, -1.5674,  0.4629,  ..., -1.9082,  3.5605,  0.6763]")
    print(f"\t10  [ 1.1055, -2.2246,  0.3535,  ..., -2.5898,  2.4023, -1.1875]")
    print(f"actual encodings:\n\tshape: {encodings.shape}\n{encodings}")
    print(f"Llama.encode_text\n\tencode in {time.time() - start_time:.2f} seconds")


llama = Llama()
llama.test_encode()
