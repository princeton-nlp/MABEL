"""
Convert MABEL's checkpoints to Huggingface style.
"""

import argparse
import torch
import os
import json
from transformers import BertForMaskedLM, RobertaForMaskedLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path of MABEL checkpoint folder")
    parser.add_argument(
        "--base-model", type=str, choices=["bert", "roberta"], default="bert"
    )
    args = parser.parse_args()

    print("MABEL checkpoint -> Huggingface checkpoint for {}".format(args.path))

    if args.base_model == "bert":
        model = BertForMaskedLM.from_pretrained(args.path)
    else:
        model = RobertaForMaskedLM.from_pretrained(args.path)

    state_dict = torch.load(
        os.path.join(args.path, "pytorch_model.bin"), map_location=torch.device("cuda")
    )

    lm_dict = {k: v for k, v in state_dict.items() if "lm_head." in k}
    new_dict = {}

    for k, v in list(lm_dict.items()):
        new_dict[k.replace("lm_head.", "")] = lm_dict.pop(k)

    if args.base_model == "bert":
        try:
            model.cls.predictions.load_state_dict(new_dict)
        except:
            raise Exception("Unable to copy LM weights over")

    torch.save(model.state_dict(), os.path.join(args.path, "pytorch_model.bin"))

    # Change architectures in config.json
    config = json.load(open(os.path.join(args.path, "config.json")))
    if args.base_model == "bert":
        for i in range(len(config["architectures"])):
            config["architectures"][i] = config["architectures"][i].replace(
                "BertForMABEL", "BertForMaskedLM"
            )
    else:
        for i in range(len(config["architectures"])):
            config["architectures"][i] = config["architectures"][i].replace(
                "RobertaForMABEL", "RobertaForMaskedLM"
            )
    json.dump(config, open(os.path.join(args.path, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
