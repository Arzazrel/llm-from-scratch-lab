# Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)", Apache 2.0. Modified for personal study.
# ---------------------------------------------------------------------------
# model_utils.py
# Utility module for the 'test' folder.
# It gathers, in a single place, all the building blocks already developed in
# the lab/ folders (GPT architecture, pretrained-weight download/loading, text
# generation, datasets, training loops) so that 'train_model.py' and
# 'test_model.py' can create, fine-tune, save, load and run the models without
# depending on the individual lab chapters.
#
# The code follows the same conventions used in the rest of the repository
# (same GPTModel, MultiHeadAttention, generate, download_and_load_gpt2, ...).
# ---------------------------------------------------------------------------

import os
import json
import urllib.request

import tiktoken
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ===========================================================================
# SECTION 1 - MODEL CONFIGURATION AND NAMING CONVENTIONS
# ===========================================================================

# Base configuration shared by every GPT-2 variant. The per-size fields
# (emb_dim, n_layers, n_heads) are added on top through MODEL_CONFIGS.
BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate (0 at inference / light finetuning)
    "qkv_bias": True,        # Query-key-value bias (needed to load GPT-2 weights)
}

# Per-size architecture (same values documented in LLM_model_done.py).
MODEL_CONFIGS = {
    "small":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "large":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "XL":     {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Mapping between the short size name used in this folder and the official
# OpenAI GPT-2 size string used by download_and_load_gpt2().
SIZE_TO_GPT2 = {
    "small":  "124M",
    "medium": "355M",
    "large":  "774M",
    "XL":     "1558M",
}

# Allowed task identifiers. The saved-model name is built from these.
#   gen                     -> simple text generation
#   adv_gen_t<val>_k<val>   -> text generation with temperature + top-k
#   spam_cls                -> spam classifier
#   sent_cls                -> sentiment analysis classifier
#   instr                   -> instruction-following finetuned model
# The full file name is always:  <task>_<size>.pth   (e.g. gen_small.pth)
TASKS = ("gen", "adv_gen", "spam_cls", "sent_cls", "instr")

# Directory that stores every trained model.
TRAINED_DIR = "trained_model"


def build_model_name(task, size, temperature=None, top_k=None):
    """Build the standardized saved-model file name (without directory).

    - gen       -> 'gen_<size>'
    - adv_gen   -> 'adv_gen_t<temperature>_k<top_k>_<size>'
    - spam_cls  -> 'spam_cls_<size>'
    - sent_cls  -> 'sent_cls_<size>'
    - instr     -> 'instr_<size>'
    The temperature/top_k values are embedded in the name only for adv_gen, so
    that test_model.py can rebuild the exact same name to look the file up.
    """
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Allowed: {list(MODEL_CONFIGS)}")

    if task == "adv_gen":
        if temperature is None or top_k is None:
            raise ValueError("adv_gen requires both temperature and top_k.")
        # format the temperature so the name stays filesystem-friendly
        t_str = _fmt_num(temperature)
        return f"adv_gen_t{t_str}_k{int(top_k)}_{size}.pth"

    if task in ("gen", "spam_cls", "sent_cls", "instr"):
        return f"{task}_{size}.pth"

    raise ValueError(f"Unknown task '{task}'. Allowed: {TASKS}")


def _fmt_num(x):
    """Format a float for use inside a file name: 1.0 -> '1', 0.7 -> '0.7'."""
    f = float(x)
    if f.is_integer():
        return str(int(f))
    return str(f).rstrip("0").rstrip(".")


def model_path(task, size, temperature=None, top_k=None, trained_dir=TRAINED_DIR):
    """Full path (trained_dir + file name) of a saved model."""
    return os.path.join(trained_dir, build_model_name(task, size, temperature, top_k))


# ===========================================================================
# SECTION 2 - TOKENIZER HELPERS (text <-> token ids)
# ===========================================================================

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)                            # remove batch dimension
    return tokenizer.decode(flat.tolist())


# ===========================================================================
# SECTION 3 - GPT ARCHITECTURE
# (identical to the implementation used across the lab, see LLM_model_done.py)
# ===========================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ===========================================================================
# SECTION 4 - MODEL BUILDERS (base model for each task + size)
# ===========================================================================

def get_config(size):
    """Return a full config dict (BASE_CONFIG + per-size fields) for a size."""
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Allowed: {list(MODEL_CONFIGS)}")
    cfg = BASE_CONFIG.copy()
    cfg.update(MODEL_CONFIGS[size])
    return cfg


def build_base_model(size):
    """Create a plain GPTModel (language-modeling head) for the given size."""
    cfg = get_config(size)
    return GPTModel(cfg)


def build_classifier_model(size, num_classes=2):
    """Create a GPTModel whose output head is replaced by a classification head.
    Used for both the spam classifier and the sentiment classifier."""
    cfg = get_config(size)
    model = GPTModel(cfg)
    model.out_head = torch.nn.Linear(in_features=cfg["emb_dim"], out_features=num_classes)
    return model


# ===========================================================================
# SECTION 5 - PRETRAINED WEIGHTS: DOWNLOAD AND LOADING
# (same code as LLM_model_done.py / gpt_generate.py)
# ===========================================================================

def download_and_load_gpt2(model_size, models_dir):
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params


def download_file(url, destination):
    with urllib.request.urlopen(url) as response:
        file_size = int(response.headers.get("Content-Length", 0))

        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        block_size = 1024
        progress_bar_description = os.path.basename(url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            with open(destination, "wb") as file:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress_bar.update(len(chunk))


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def load_pretrained_into_model(model, size, models_dir="gpt2"):
    """Download the GPT-2 weights for 'size' and load them into 'model'."""
    gpt2_size = SIZE_TO_GPT2[size]
    settings, params = download_and_load_gpt2(model_size=gpt2_size, models_dir=models_dir)
    load_weights_into_gpt(model, params)
    return model


# ===========================================================================
# SECTION 6 - TEXT GENERATION
# ===========================================================================

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedy decoding: always pick the most likely next token."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Advanced decoding with optional temperature scaling and top-k sampling."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ===========================================================================
# SECTION 7 - SAVE / LOAD TRAINED MODELS
# ===========================================================================

def save_model(model, task, size, temperature=None, top_k=None, trained_dir=TRAINED_DIR):
    """Save a trained model using the standardized naming convention."""
    os.makedirs(trained_dir, exist_ok=True)
    path = model_path(task, size, temperature, top_k, trained_dir)
    torch.save(model.state_dict(), path)
    print(f"Model saved as: {path}")
    return path


def load_model(path, model, device):
    """Load a saved state_dict into an already-built model (same as the lab)."""
    model_state_dict = torch.load(path, map_location=device)
    model.load_state_dict(model_state_dict)
    return model


def list_trained_models(trained_dir=TRAINED_DIR):
    """Return the list of .pth files currently present in trained_dir."""
    if not os.path.isdir(trained_dir):
        return []
    return sorted(f for f in os.listdir(trained_dir) if f.endswith(".pth"))


# ===========================================================================
# SECTION 8 - CLASSIFICATION: DATASET, TRAINING, INFERENCE
# (spam and sentiment share the same machinery; only data source differs)
# ===========================================================================

class ClassificationDataset(Dataset):
    """Generic CSV-backed dataset for single-text classification.
    Expects two columns whose names are passed in (text_col, label_col).
    Mirrors SpamDataset / IMDBDataset from the lab."""
    def __init__(self, csv_file, tokenizer, text_col, label_col, max_length=None, pad_token_id=50256):
        import pandas as pd
        self.data = pd.read_csv(csv_file)
        self.text_col = text_col
        self.label_col = label_col
        self.encoded_texts = [tokenizer.encode(str(t)) for t in self.data[text_col]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [et[:self.max_length] for et in self.encoded_texts]

        # pad to the longest sequence
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index][self.label_col]
        return (torch.tensor(encoded, dtype=torch.long),
                torch.tensor(int(label), dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max((len(et) for et in self.encoded_texts), default=0)


def calc_loss_batch_cls(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]                 # logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader_cls(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch_cls(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def evaluate_model_cls(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader_cls(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader_cls(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):
    """Classification finetuning loop (same logic as the lab version)."""
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch_cls(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_cls(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | "
              f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_text(text, model, tokenizer, device, label_map, max_length=None, pad_token_id=50256):
    """Run a single-text classification and map the predicted class to a label.
    label_map example: {0: 'not spam', 1: 'spam'} or {0: 'negative', 1: 'positive'}."""
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    if max_length is None:
        max_length = len(input_ids)
    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return label_map.get(predicted_label, str(predicted_label))


# ===========================================================================
# SECTION 9 - INSTRUCTION FINETUNING: FORMAT, DATASET, COLLATE, TRAIN
# (same Alpaca-style formatting and masking as the book)
# ===========================================================================

def format_input(entry):
    """Alpaca-style prompt formatting (instruction + optional input)."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    """Pad a batch, build shifted targets, and mask padding in the loss.

    'mask_all_padding' behaviour follows the book: every padding token after
    the first one is replaced by ignore_index so it does not contribute to the
    loss. The variant that also masks the instruction part is handled upstream
    (see train_model.py, option 'mask_instruction')."""
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def mask_instruction_in_targets(inputs_tensor, targets_tensor, tokenizer, ignore_index=-100):
    """Optional variant: also mask the instruction/input part of each sample so
    the loss is computed only on the response tokens.
    It locates the '### Response:' marker and sets every target before it to
    ignore_index. Returns a new targets tensor."""
    response_marker = tokenizer.encode("\n\n### Response:\n")
    marker_len = len(response_marker)
    new_targets = targets_tensor.clone()

    # inputs_tensor is already shifted by -1 w.r.t. targets in custom_collate_fn,
    # so we search the marker on the input row and mask the aligned target cells.
    for row in range(inputs_tensor.shape[0]):
        ids = inputs_tensor[row].tolist()
        cut = None
        for i in range(len(ids) - marker_len + 1):
            if ids[i:i + marker_len] == response_marker:
                cut = i + marker_len
                break
        if cut is not None:
            new_targets[row, :cut] = ignore_index
    return new_targets


def calc_loss_batch_lm(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader_lm(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch_lm(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model_lm(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader_lm(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader_lm(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, tokenizer):
    """Language-modeling / instruction finetuning loop (same logic as the lab)."""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch_lm(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_lm(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

    return train_losses, val_losses, track_tokens_seen
