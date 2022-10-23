"""
Convert MABEL's checkpoints to Huggingface style.
"""

import argparse
import torch
import os
import json
from transformers import BertForMaskedLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path of MABEL checkpoint folder")
    args = parser.parse_args()

    print("MABEL checkpoint -> Huggingface checkpoint for {}".format(args.path))

    state_dict = torch.load(
        os.path.join(args.path, "pytorch_model.bin"), map_location=torch.device("cuda")
    )

    new_state_dict = {}
    for key, param in state_dict.items():
        # Replace "mlp" to "pooler"
        if "mlp" in key:
            key = key.replace("mlp", "pooler")

        # Delete "bert" or "roberta" prefix
        if "embeddings." in key and "_embeddings" not in key:
            key = key.replace("embeddings.", "bert.embeddings.")
        if "encoder." in key:
            key = key.replace("encoder.", "bert.encoder.")
        if "embeddings.word_bert." in key:
            key = key.replace(
                "embeddings.word_bert.", "bert.embeddings.word_embeddings."
            )
        if "embeddings.position_bert." in key:
            key = key.replace(
                "embeddings.position_bert.", "bert.embeddings.position_embeddings."
            )
        if "embeddings.token_type_bert." in key:
            key = key.replace(
                "embeddings.token_type_bert.", "bert.embeddings.token_type_embeddings."
            )

        if "embeddings.word_embeddings.weight" == key:
            key = "bert.embeddings.word_embeddings.weight"
        if "embeddings.position_embeddings.weight" == key:
            key = "bert.embeddings.position_embeddings.weight"
        if "embeddings.token_type_embeddings.weight" == key:
            key = "bert.embeddings.token_type_embeddings.weight"

        if "lm_head." in key:
            key = key.replace("lm_head.", "cls.predictions.")

        new_state_dict[key] = param

    torch.save(new_state_dict, os.path.join(args.path, "pytorch_model.bin"))

    # Change architectures in config.json
    config = json.load(open(os.path.join(args.path, "config.json")))
    for i in range(len(config["architectures"])):
        config["architectures"][i] = config["architectures"][i].replace(
            "BertForMABEL", "BertForMaskedLM"
        )
    json.dump(config, open(os.path.join(args.path, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
