import argparse
from argparse import Namespace

import logging
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from model import LinearClassifier

from datasets import load_dataset

from transformers import AutoTokenizer, AutoConfig


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataparallel", default=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--tau2", type=float, default=0.7)
    parser.add_argument("--cache_dir", default=None, type=str, help="cache directory")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--eval_data_path", type=str, default="bias-nli.csv")
    parser.add_argument(
        "--load_from_file",
        type=str,
        default=None,
        help="path to evaluation .pt checkpoint",
    )

    args = parser.parse_args()
    args.update_encoder = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=3, output_hidden_states=True
    )

    model = LinearClassifier(config=config, args=args).to(args.device)
    states = torch.load(args.load_from_file, map_location=torch.device(args.device))[
        "states"
    ]
    model.load_state_dict(states)
    model.eval()

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples["premise"],)
            if "hypothesis" == None
            else (examples["premise"], examples["hypothesis"])
        )
        result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)
        return result

    eval_dataset = load_dataset(
        "csv",
        data_files=args.eval_data_path,
        split="train[:-10%]",
        cache_dir=args.cache_dir,
    )
    eval_dataset = eval_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.map(
        preprocess_function, batched=True, load_from_cache_file=True
    )

    logger.info(f"Number of examples: {len(eval_dataset)}")
    eval_loader = DataLoader(
        dataset=eval_dataset, batch_size=args.batch_size, shuffle=True
    )

    nn_count, fn_count, tn_count, tn2_count, denom = 0, 0, 0, 0, 0

    for batch_idx, batch in enumerate(tqdm(eval_loader)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(
            args.device
        )
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(
            args.device
        )

        labels = torch.tensor(batch["label"]).long().to(args.device)

        if "token_type_ids" in batch:
            token_type_ids = torch.transpose(
                torch.stack(batch["token_type_ids"]), 0, 1
            ).to(args.device)
            output = model.forward_eval(
                input_ids, attention_mask, token_type_ids, labels
            )
        else:
            output = model.forward_eval(input_ids, attention_mask, None, labels)

        res = torch.softmax(output.logits, axis=1)
        preds = res.argmax(1)
        denom += len(preds)

        nn_count += (torch.sum(res, axis=0)[1]).item()
        fn_count += (torch.bincount(preds)[1]).item()
        tn_count += torch.sum(res[:, 1] > args.tau).item()
        tn2_count += torch.sum(res[:, 1] > args.tau2).item()

        if batch_idx % args.print_every == 0:
            logger.info(f"net neutral: {nn_count / denom}")
            logger.info(f"fraction neutral: {fn_count / denom}")
            logger.info(f"tau 0.5 neutral: {tn_count / denom}")
            logger.info(f"tau 0.7 neutral: {tn2_count / denom}")

    logger.info(f"total net neutral: {nn_count / denom}")
    logger.info(f"total fraction neutral: {fn_count / denom}")
    logger.info(f"total tau 0.5 neutral: {tn_count / denom}")
    logger.info(f"total tau 0.7 neutral: {tn2_count / denom}")


if __name__ == "__main__":
    main()
