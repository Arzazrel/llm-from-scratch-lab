# Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)", Apache 2.0. Modified for personal study.
# ---------------------------------------------------------------------------
# test_model.py
# Interactive runner for the trained models saved in 'trained_model/'.
# It extends the idea of spam_classifier_test.py / sentiment_classifier_test.py
# to every task of the repository.
#
# Flow:
#   1) the user chooses the functionality (task);
#   2) for text generation, the relevant decoding parameters are asked
#      (window size; plus temperature and top-k for the advanced variant),
#      each with a sensible default;
#   3) the user chooses the GPT-2 size (small/medium/large/XL);
#   4) the matching trained model is looked up in 'trained_model/':
#        - if missing  -> print an error, list all available models, exit;
#        - if present   -> load it, print a success message;
#   5) the program asks for an input, shows the output, then asks whether to
#      continue with another input or quit.
# ---------------------------------------------------------------------------

import os
import torch

import model_utils as mu


# ---------------------------------------------------------------------------
# Small input helpers (defaults on empty input)
# ---------------------------------------------------------------------------

def ask_choice(prompt, choices, default):
    choices_str = "/".join(choices)
    while True:
        raw = input(f"{prompt} [{choices_str}] (default: {default}): ").strip()
        if raw == "":
            return default
        if raw in choices:
            return raw
        print(f"  Invalid choice. Allowed: {choices_str}")


def ask_int(prompt, default):
    raw = input(f"{prompt} (default: {default}): ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  Not an integer, using default {default}.")
        return default


def ask_float(prompt, default):
    raw = input(f"{prompt} (default: {default}): ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Not a number, using default {default}.")
        return default


def wants_to_continue():
    raw = input("\nDo you want to continue? [c = continue / q = quit] (default: c): ").strip().lower()
    return raw in ("", "c", "continue", "y", "yes")


# ---------------------------------------------------------------------------
# Default decoding parameters (aligned with the lab files)
# ---------------------------------------------------------------------------

DEFAULT_MAX_NEW_TOKENS = 50     # generation window; realistic small value for GPT-2 demos
DEFAULT_TEMPERATURE = 1.0       # 1.0 = unscaled distribution (as in the lab examples)
DEFAULT_TOP_K = 5               # top-k in line with the lab (examples used 3 or 5)


# ---------------------------------------------------------------------------
# Loading: find and load the matching trained model, or exit with the list
# ---------------------------------------------------------------------------

def load_or_exit(task, size, model, device, temperature=None, top_k=None):
    """Look up the standardized file name; load it or exit listing what exists."""
    path = mu.model_path(task, size, temperature=temperature, top_k=top_k)
    if not os.path.exists(path):
        print(f"\n[ERROR] Trained model not found: {path}")
        available = mu.list_trained_models()
        if available:
            print("Available trained models in "
                  f"'{mu.TRAINED_DIR}/':")
            for name in available:
                print("   -", name)
        else:
            print(f"No trained models found in '{mu.TRAINED_DIR}/'. "
                  f"Run train_model.py first.")
        raise SystemExit(1)

    mu.load_model(path, model, device)
    model.to(device)
    print(f"\n[OK] Model loaded successfully: {path}")
    return model


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_simple_generation(size, device, tokenizer):
    max_new_tokens = ask_int("Generation window (number of new tokens)", DEFAULT_MAX_NEW_TOKENS)

    model = mu.build_base_model(size)
    load_or_exit("gen", size, model, device)
    context_size = mu.get_config(size)["context_length"]

    print("\n--- Simple text generation (greedy decoding) ---")
    while True:
        prompt = input("\nEnter your prompt: ")
        idx = mu.text_to_token_ids(prompt, tokenizer).to(device)
        out = mu.generate_text_simple(model, idx, max_new_tokens=max_new_tokens,
                                      context_size=context_size)
        text = mu.token_ids_to_text(out, tokenizer)
        print("\nInput :", prompt)
        print("Output:", text)
        if not wants_to_continue():
            break


def run_advanced_generation(size, device, tokenizer):
    max_new_tokens = ask_int("Generation window (number of new tokens)", DEFAULT_MAX_NEW_TOKENS)
    temperature = ask_float("Temperature", DEFAULT_TEMPERATURE)
    top_k = ask_int("Top-k", DEFAULT_TOP_K)

    model = mu.build_base_model(size)
    # The advanced model shares the same weights as 'gen'; the name just encodes
    # the preset. If an adv_gen file with these exact params is missing, we fall
    # back to a plain 'gen' model of the same size so generation still works.
    path = mu.model_path("adv_gen", size, temperature=temperature, top_k=top_k)
    if os.path.exists(path):
        mu.load_model(path, model, device)
        model.to(device)
        print(f"\n[OK] Model loaded successfully: {path}")
    else:
        print(f"\n[WARN] {path} not found, trying the plain generation model instead.")
        load_or_exit("gen", size, model, device)

    context_size = mu.get_config(size)["context_length"]

    print("\n--- Advanced text generation (temperature + top-k) ---")
    print(f"Parameters: temperature={temperature}, top_k={top_k}, window={max_new_tokens}")
    while True:
        prompt = input("\nEnter your prompt: ")
        idx = mu.text_to_token_ids(prompt, tokenizer).to(device)
        out = mu.generate(model, idx, max_new_tokens=max_new_tokens,
                          context_size=context_size, temperature=temperature,
                          top_k=top_k, eos_id=50256)
        text = mu.token_ids_to_text(out, tokenizer)
        print("\nInput :", prompt)
        print("Output:", text)
        if not wants_to_continue():
            break


def run_classification(task, size, device, tokenizer):
    label_map = ({0: "not spam", 1: "spam"} if task == "spam_cls"
                 else {0: "negative", 1: "positive"})
    title = "Spam classification" if task == "spam_cls" else "Sentiment analysis"

    model = mu.build_classifier_model(size, num_classes=2)
    load_or_exit(task, size, model, device)

    print(f"\n--- {title} ---")
    while True:
        text = input("\nEnter the text to classify: ")
        result = mu.classify_text(text, model, tokenizer, device, label_map,
                                  max_length=len(text))
        print("\nInput :", text)
        print("Output:", result)
        if not wants_to_continue():
            break


def run_instruction(size, device, tokenizer):
    max_new_tokens = ask_int("Response window (number of new tokens)", 256)

    model = mu.build_base_model(size)
    load_or_exit("instr", size, model, device)
    context_size = mu.get_config(size)["context_length"]

    print("\n--- Instruction following ---")
    print("Tip: just type the instruction. If you need to provide an extra input,")
    print("     you will be asked for it after the instruction (leave empty to skip).")
    while True:
        instruction = input("\nEnter your instruction: ")
        extra = input("Optional input (press Enter to skip): ")
        entry = {"instruction": instruction, "input": extra}
        prompt = mu.format_input(entry)

        idx = mu.text_to_token_ids(prompt, tokenizer).to(device)
        out = mu.generate(model, idx, max_new_tokens=max_new_tokens,
                          context_size=context_size, eos_id=50256)
        full = mu.token_ids_to_text(out, tokenizer)
        response = full[len(prompt):].replace("### Response:", "").strip()

        print("\nInput :", instruction if not extra else f"{instruction} | {extra}")
        print("Output:", response)
        if not wants_to_continue():
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_MENU = {
    "1": ("Simple text generation", "gen"),
    "2": ("Text generation with temperature + top-k", "adv_gen"),
    "3": ("Spam classification", "spam_cls"),
    "4": ("Sentiment analysis", "sent_cls"),
    "5": ("Instruction following", "instr"),
}


def main():
    print("=" * 60)
    print(" GPT-2 model tester")
    print("=" * 60)
    print("Which functionality do you want to try?")
    for key, (label, _) in TASK_MENU.items():
        print(f"  {key}) {label}")

    choice = ask_choice("Choose", list(TASK_MENU.keys()), "1")
    _, task = TASK_MENU[choice]

    size = ask_choice("\nWhich GPT-2 model size?", ["small", "medium", "large", "XL"], "small")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = mu.get_tokenizer()
    print("Device:", device)

    if task == "gen":
        run_simple_generation(size, device, tokenizer)
    elif task == "adv_gen":
        run_advanced_generation(size, device, tokenizer)
    elif task in ("spam_cls", "sent_cls"):
        run_classification(task, size, device, tokenizer)
    elif task == "instr":
        run_instruction(size, device, tokenizer)

    print("\nGoodbye.")


if __name__ == "__main__":
    main()
