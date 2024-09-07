import transformers

from config import GPT2Config
from model import GPT2
import torch
import os
import requests
import numpy as np
import torch.nn.functional as F
from transformers import GPT2Tokenizer

def download_weights():
    # load pretrained_weights from hugging face
    url = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin"
    filename = "gpt2-pytorch_model.bin"
    directory = "./weights"

    os.makedirs(directory, exist_ok=True)

    response = requests.get(url)

    if response.status_code == 200:

        file_path = os.path.join(directory, filename)

        with open(file_path, 'wb') as file:
            file.write(response.content)

        print(f"File downloaded and saved to: {file_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def load_pretrained(model: GPT2):
    if not os.path.exists("./weights/gpt2-pytorch_model.bin"):
        download_weights()

    model_dict = model.state_dict()  # currently with random initialization
    state_dict = torch.load("./weights/gpt2-pytorch_model.bin")  # pretrained weights

    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        if "mlp" in key:  # The hugging face state dict references the feedforward network as mlp, need to replace to `feedforward` be able to reuse these weights
            new_key = key.replace("mlp", "feedforward")
            new_keys.append(new_key)
            old_keys.append(key)

    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def load_model(use_pretrained: bool = True):
    config = GPT2Config()
    model = GPT2(config)

    if use_pretrained:
        model = load_pretrained(model)

    model.eval()  # model in inference mode

    return model

def generate(context, ntok=20):
    for _ in range(ntok):
        out = model(context)
        logits = out[:, -1, :]
        indices_to_remove = logits < torch.topk(logits, 10)[0][..., -1, None]
        logits[indices_to_remove] = np.NINF
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
        context = torch.cat([context, next_tok.unsqueeze(-1)], dim=-1)

    return context


if __name__ == "__main__":

    transformers.utils.move_cache()

    model = load_model(use_pretrained=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    context = torch.tensor([tokenizer.encode("The planet earth")])

    out = generate(context, ntok=20)
    print(tokenizer.decode(out[0]))

