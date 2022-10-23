import json
import logging
import os
import random
from time import time
import torch.distributed
from pathlib import Path
import argparse
import pickle

import yaml
import glob
import logging
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW

logger = logging.getLogger(__name__)

with open("prof2ind.json") as json_file:
    mapping = json.load(json_file)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def save_ckpt(model, optimizer, args, latest=False):
    if not latest:
        best_ckpt_path = args.file_path.format(
            step=args.global_step, best_score=args.best_score * 100
        )
        checkpoint = {"ckpt_path": best_ckpt_path}
    else:
        checkpoint = {"ckpt_path": os.path.join(args.ckpt_dir, "latest.ckpt")}

    states = model.state_dict() if not args.dataparallel else model.module.state_dict()
    checkpoint["states"] = states
    checkpoint["optimizer_states"] = optimizer.state_dict()

    if not latest:
        for rm_path in glob.glob(os.path.join(args.ckpt_dir, "*.pt")):
            os.remove(rm_path)

    torch.save(checkpoint, checkpoint["ckpt_path"])
    print(f"Model saved at: {checkpoint['ckpt_path']}")


def save_args(args):
    with open(args.args_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Arg file saved at: {args.args_path}")


def process_data(dataset):
    occ = []
    bio = []
    gend = []

    for elem in dataset:
        occ.append(elem["title"])
        bio.append(elem["raw"][elem["start_pos"] :])
        gend.append(elem["gender"])

    prof_result = []

    for _, v in enumerate(occ):
        try:
            index = mapping[v]
        except KeyError:
            raise Exception("unknown label in occupation")
        prof_result.append(index)

    gend_result = []

    for _, v in enumerate(gend):
        if v == "m":
            gend_result.append(0)
        elif v == "f":
            gend_result.append(1)
        else:
            raise Exception("unknown label in gender")

    data_dict = {"label": prof_result, "bio": bio, "gend": gend_result}
    dataset = Dataset.from_dict(data_dict)
    return dataset


def train_epoch(epoch, model, dl, eval_dl, optimizer, args):
    logger.info(f"At epoch {epoch}:")
    model.train()

    t = time()
    total_loss = 0
    total_correct = []

    for batch_idx, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(
            args.device
        )
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(
            args.device
        )
        labels = torch.tensor(batch["label"]).long().to(args.device)

        output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        batch_loss = output.loss
        logits = output.logits

        if args.dataparallel:
            total_loss += batch_loss.mean().item()
            batch_loss.mean().backward()
        else:
            total_loss += batch_loss.item()
            batch_loss.backward()

        total_correct += (torch.argmax(logits, dim=1) == labels).tolist()

        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()

        if args.global_step % args.print_every == 0:
            train_acc = sum(total_correct) / len(total_correct)
            train_loss = total_loss / len(total_correct)
            logger.info(f"train_acc: {train_acc}, train_loss: {train_loss}")

        if args.global_step % args.eval_interval == 0 and args.global_step != 0:
            dev_acc = evaluate("dev", model, eval_dl, epoch, args)
            if dev_acc > args.best_score:
                args.best_score = dev_acc
            save_ckpt(model, optimizer, args, latest=False)

        args.global_step += 1


def evaluate(mode, model, dl, epoch, args):
    model.eval()
    logger.info("running eval")

    total_loss = 0
    total_correct = []
    for batch_idx, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(
            args.device
        )
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(
            args.device
        )
        labels = torch.tensor(batch["label"]).long().to(args.device)

        batch_loss = None
        logits = None

        with torch.no_grad():
            if args.dataparallel:
                output = model.module.forward(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
            else:
                output = model.forward(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

        batch_loss = output.loss
        logits = output.logits

        if args.dataparallel:
            total_loss += batch_loss.sum().item()
        else:
            total_loss += batch_loss.item()
        total_correct += (torch.argmax(logits, dim=1) == labels).tolist()

    acc = sum(total_correct) / len(total_correct)
    loss = total_loss / len(total_correct)
    log_dict = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "eval_acc": acc,
        "eval_loss": loss,
        "global_step": args.global_step,
    }
    logger.info(mode, log_dict)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataparallel", default=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--print_every", default=50, type=int)
    parser.add_argument("--eval_interval", default=1000, type=int)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--cache_dir", default=None, type=str, help="cache directory")
    parser.add_argument(
        "--file_name", default="model-step={step}-acc={best_score:.2f}.pt"
    )
    parser.add_argument("--train_path", type=str, default="trainbios.pkl")
    parser.add_argument("--val_path", type=str, default="valbios.pkl")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument(
        "--max_seq_len", default=128, type=int, help="Max. sequence length"
    )
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument(
        "--fix_encoder",
        action="store_true",
        help="Whether to fix encoder - default false.",
    )

    args = parser.parse_args()
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    args.best_score = float("-inf")
    args.global_step = 0
    args.args_path = os.path.join(args.ckpt_dir, "args.yaml")
    args.file_path = os.path.join(args.ckpt_dir, args.file_name)
    save_args(args)

    train_file = open(args.train_path, "rb")
    train_data = pickle.load(train_file)
    train_file.close()

    val_file = open(args.train_path, "rb")
    val_data = pickle.load(val_file)
    val_file.close()

    train_dataset = process_data(train_data)
    val_dataset = process_data(val_data)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, cache_dir=args.cache_dir
    )

    def preprocess_function(examples):
        # Tokenize the texts
        args = [examples["bio"]]
        result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)
        return result

    train_dataset = train_dataset.map(
        preprocess_function, batched=True, load_from_cache_file=True
    )
    val_dataset = val_dataset.map(
        preprocess_function, batched=True, load_from_cache_file=True
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(mapping),
        cache_dir=args.cache_dir,
        output_hidden_states=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir, config=config
    )

    if args.fix_encoder:
        print("FIXING ENCODER...")
        if "roberta" in args.model_name_or_path:
            for param in model.roberta.parameters():
                param.requires_grad = False
        else:
            for param in model.bert.parameters():
                param.requires_grad = False

    if torch.cuda.device_count() and args.dataparallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model = model.to(args.device)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.warning(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=False
    )
    eval_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False
    )

    if args.dataparallel:
        logger.info(
            f"Trainable params: {sum(p.numel() for p in model.module.parameters() if p.requires_grad)}"
        )
        logger.info(
            f"All params      : {sum(p.numel() for p in model.module.parameters())}"
        )
    else:
        logger.info(
            f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        logger.info(f"All params      : {sum(p.numel() for p in model.parameters())}")

    if args.dataparallel:
        optimizer = AdamW(model.module.parameters(), lr=args.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(int(args.num_epochs)):
        train_epoch(epoch, model, train_loader, eval_loader, optimizer, args)

    save_ckpt(model, optimizer, args, latest=True)
    logger.info(f"Finished training with highest dev acc. {args.best_score}")


if __name__ == "__main__":
    main()
