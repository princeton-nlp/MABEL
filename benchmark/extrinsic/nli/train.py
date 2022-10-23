import logging
import os
import random
from dataclasses import dataclass, field
from time import time
import torch.distributed
from datasets import load_dataset

from pathlib import Path
import shutil

import argparse

import yaml
import glob

import torch


from model import LinearClassifier

from transformers import AutoTokenizer, AutoConfig

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from torch.utils.data import DataLoader

from tqdm import tqdm

logger = logging.getLogger(__name__)

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


def evaluate(mode, model, dl, epoch, args):
    model.eval()
    logger.info(f"Evaluating at {args.global_step}")

    total_loss = 0
    total_correct = []
    for batch_idx, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(
            args.device
        )
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(
            args.device
        )
        if "token_type_ids" in batch:
            token_type_ids = torch.transpose(
                torch.stack(batch["token_type_ids"]), 0, 1
            ).to(args.device)
        else:
            token_type_ids = None
        labels = torch.tensor(batch["label"]).long().to(args.device)

        batch_loss = None
        logits = None

        if args.dataparallel:
            output = model.module.forward(
                input_ids, attention_mask, token_type_ids, labels=labels
            )
        else:
            output = model.forward(
                input_ids, attention_mask, token_type_ids, labels=labels
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
    print(mode, log_dict)
    return acc


def train_epoch(epoch, model, dl, eval_dl, optimizer, scheduler, args):
    print(f"At epoch {epoch}:")
    model.train()

    t = time()
    total_loss = 0
    total_correct = []

    for _, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(
            args.device
        )
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(
            args.device
        )
        if "token_type_ids" in batch:
            token_type_ids = torch.transpose(
                torch.stack(batch["token_type_ids"]), 0, 1
            ).to(args.device)
        else:
            token_type_ids = None
        labels = torch.tensor(batch["label"]).long().to(args.device)

        output = model(input_ids, attention_mask, token_type_ids, labels=labels)
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
        scheduler.step()

        if args.global_step % args.print_every == 0:
            train_acc = sum(total_correct) / len(total_correct)
            train_loss = total_loss / len(total_correct)
            print(f"train_acc: {train_acc}, train_loss: {train_loss}")

        if args.global_step % args.eval_interval == 0 and args.global_step != 0:
            dev_acc = evaluate("dev", model, eval_dl, epoch, args)
            if dev_acc > args.best_score:
                args.best_score = dev_acc
                save_ckpt(model, optimizer, args, latest=False)
        args.global_step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataparallel", default=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--ckpt_dir", type=str, default="nli-ckpts")
    parser.add_argument(
        "--cache_dir", type=str, default=".cache", help="cache directory"
    )
    parser.add_argument(
        "--file_name", default="model-step={step}-acc={best_score:.2f}.pt"
    )
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--load_from_file", type=str, default=None)
    parser.add_argument(
        "--fix_encoder",
        action="store_true",
        help="whether or not to update encoder; default False",
    )
    parser.add_argument("--eval_dataset_path", type=str, default="bias-nli.csv")

    args = parser.parse_args()
    if os.path.exists(args.ckpt_dir):
        shutil.rmtree(args.ckpt_dir)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    args.best_score = float("-inf")
    args.global_step = 0
    args.file_path = os.path.join(args.ckpt_dir, args.file_name)
    args.args_path = os.path.join(args.ckpt_dir, "args.yaml")

    logger.info(args.ckpt_dir)

    save_args(args)

    datasets = load_dataset("snli", cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=3, output_hidden_states=True
    )

    model = LinearClassifier(config=config, args=args).to(args.device)

    if args.fix_encoder:
        logger.info("Fixing encoder params")
        for param in model.model.parameters():
            param.requires_grad = False

    if torch.cuda.device_count() and args.dataparallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model = model.to(args.device)

    def preprocess_function(examples):
        # Tokenize the texts
        args = [examples["premise"], examples["hypothesis"]]
        result = tokenizer(*args, padding="max_length", max_length=64, truncation=True)
        return result

    datasets = datasets.map(
        preprocess_function, batched=True, load_from_cache_file=True
    )

    train_dataset = datasets["train"]
    train_dataset = train_dataset.filter(lambda x: x["label"] != -1)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.warning(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=False
    )

    eval_dataset = load_dataset(
        "csv",
        data_files=args.eval_dataset_path,
        split="train[:5%]",
        cache_dir=args.cache_dir,
    )
    eval_dataset = eval_dataset.shuffle(seed=args.random_seed)
    eval_dataset = eval_dataset.map(
        preprocess_function, batched=True, load_from_cache_file=True
    )
    logger.info(f"Number of examples: {len(eval_dataset)}")
    eval_loader = DataLoader(
        dataset=eval_dataset, batch_size=args.batch_size, shuffle=False
    )

    if args.dataparallel:
        print(
            f"Trainable params: {sum(p.numel() for p in model.module.parameters() if p.requires_grad)}"
        )
        print(f"All params      : {sum(p.numel() for p in model.module.parameters())}")
    else:
        print(
            f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        print(f"All params      : {sum(p.numel() for p in model.parameters())}")

    if args.dataparallel:
        optimizer = AdamW(
            (p for p in model.module.parameters() if p.requires_grad), lr=args.lr
        )
    else:
        optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=args.lr
        )

    scheduler = LinearLR(optimizer)

    for epoch in range(int(args.num_epochs)):
        train_epoch(epoch, model, train_loader, eval_loader, optimizer, scheduler, args)

    save_ckpt(model, optimizer, args, latest=True)
    logger.info(f"Finished training with highest dev acc. {args.best_score}")


if __name__ == "__main__":
    main()
