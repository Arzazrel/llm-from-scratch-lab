# FLOPS Analysis
# - FLOPs (Floating Point Operations Per Second) measure the computational complexity of neural network models by counting the number of floating-point operations executed
# - High FLOPs indicate more intensive computation and energy consumption

import importlib
import importlib.util
from importlib.metadata import version
import matplotlib
import torch
from thop import profile
# import of local module
spec = importlib.util.spec_from_file_location(
    "GPTModel", 
    "2_coding_an_llm_architecture.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

print("Libraary version:")
print("thop version:", version("thop"))
print("torch version:", version("torch"))

print("We measure the computational complexity (in FLOPs) and the number of parameters for various versions of a GPT-style model, using the thop (PyTorch-OpCounter) library.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("This script use this device: ", "gpu-cuda" if torch.cuda.is_available() else "cpu")

# -- define the GPT-model configuration --
BASE_CONFIG = {
    "vocab_size": 50257,     # vocabulary size
    "context_length": 1024,  # context length
    "drop_rate": 0.0,        # dropout rate [0,1]
    "qkv_bias": True         # Query-Key-Value bias in the multi-head attention mechanism when computing query (Q), key (K), and value (V) tensors
}

# values of the parameters that change across the version model
model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# -- iterate and compute the FLPOS --

input_tensor = torch.randint(0, 50257, (2, 1024)).to(device)    # create the input

print("\nNow analyze the model versions:")
for size in model_configs:
    BASE_CONFIG.update(model_configs[size])             # create the current version of the model
    print("\nCurrent version:",BASE_CONFIG)
    
    model = module.GPTModel(BASE_CONFIG).bfloat16()     # convert the weight in Brain Floating Point (standard for LLM)
    model.to(device)

    # performs profiling of the neural network. Given an input, it counts how many mathematical operations are executed within the network, layer by layer.
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)    # MACS = multiply-accumulate operations
    flops = 2 * macs                                                        # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    print(f"{size:18}: {flops:.1e} FLOPS")
    
    # clean VRAM
    del model
    torch.cuda.empty_cache()