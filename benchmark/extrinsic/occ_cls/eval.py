import json
import logging
import random
from dataclasses import dataclass, field
from time import time
import torch.distributed
import numpy as np

from collections import defaultdict, Counter
import argparse
import pickle
import logging

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

from torch.utils.data import DataLoader

from tqdm import tqdm
import torch.nn as nn


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

with open("gend2ind.json") as json_file:
    mapping = json.load(json_file)


def process_data(dataset):
    occ = []
    bio = []
    gend = []

    for elem in dataset:
        occ.append(elem["title"])
        bio.append(elem["raw"][elem["start_pos"] :])
        gend.append(elem["gender"])

    result = []

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


def rms_diff(tpr_diff):
    return np.sqrt(np.mean(tpr_diff**2))


def eval_model(model, dl, args):
    model.eval()
    print("running eval")

    total_loss = 0
    total_correct = []
    total_gender = []
    total_occ = []
    m_count = 0
    f_count = 0
    m_tot = 0
    f_tot = 0

    for batch_idx, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(
            args.device
        )

        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(
            args.device
        )
        labels = torch.tensor(batch["label"]).long().to(args.device)
        gender = batch["gend"]

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
        total_gender.append(gender)
        total_occ.append(labels.cpu().numpy())

    total_gender = [item for sublist in total_gender for item in sublist]
    total_occ = [item for sublist in total_occ for item in sublist]

    scores = defaultdict(Counter)
    prof_count_total = defaultdict(Counter)

    for (g, oc, l) in zip(total_gender, total_occ, total_correct):
        if g == "F":
            if l:
                f_count += 1
            f_tot += 1
        else:
            assert g == "M"
            if l:
                m_count += 1
            m_tot += 1

        if l is True:
            scores[oc][g] += 1
        prof_count_total[oc][g] += 1

    print(m_count)
    print(f_count)
    print(sum(total_correct))

    assert m_count + f_count == sum(total_correct)
    acc = sum(total_correct) / len(total_correct)
    m_acc = m_count / m_tot
    f_acc = f_count / f_tot

    tprs = defaultdict(dict)
    tprs_change = dict()

    for profession, scores_dict in scores.items():
        good_m, good_f = scores_dict["M"], scores_dict["F"]
        prof_total_f = prof_count_total[profession]["F"]
        prof_total_m = prof_count_total[profession]["M"]
        tpr_m = (good_m) / prof_total_m
        tpr_f = (good_f) / prof_total_f

        tprs[profession]["M"] = tpr_m
        tprs[profession]["F"] = tpr_f
        tprs_change[profession] = tpr_m - tpr_f

    tpr_rms = rms_diff(np.array(list(tprs_change.values())))

    loss = total_loss / len(total_correct)
    log_dict = {
        "batch_idx": batch_idx,
        "eval_acc": acc,
        "eval_acc_m": m_acc,
        "eval_acc_f": f_acc,
        "tpr": m_acc - f_acc,
        "tpr_rms": tpr_rms,
        "eval_loss": loss,
    }

    return log_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataparallel", default=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--cache_dir", default=None, type=str, help="cache directory")
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_path", type=str, default="testbios.pkl")
    parser.add_argument(
        "--load_from_file",
        type=str,
        help="fine-tuned checkpoint to evaluate",
        required=True,
    )

    args = parser.parse_args()
    file = open(args.test_path, "rb")
    data = pickle.load(file)
    file.close()
    _, _, test_dataset = process_data(data)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, cache_dir=args.cache_dir
    )
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=28,
        cache_dir=args.cache_dir,
        output_hidden_states=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.load_from_file, cache_dir=args.cache_dir, config=config
    )

    if args.dataparallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model = model.to(args.device)

    model.eval()

    def preprocess_function(examples):
        # Tokenize the texts
        args = [examples["bio"]]
        result = tokenizer(
            *args,
            padding="max_length",
            max_length=args.max_sequence_length,
            truncation=True,
        )
        return result

    test_dataset = test_dataset.map(
        preprocess_function, batched=True, load_from_cache_file=True
    )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(test_dataset)), 1):
        logger.warning(f"Sample {index} of the training set: {test_dataset[index]}.")

    eval_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
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

    log_dict = eval_model(model, eval_loader, args)
    print("Bias-in-Bios evaluation results:")
    print(f" - model checkpoint: {args.load_from_file}")
    print(f" - acc. (all): {log_dict['eval_acc']*100}")
    print(f" - acc. (m): {log_dict['eval_acc_m']*100}")
    print(f" - acc. (f): {log_dict['eval_acc_f']*100}")
    print(f" - tpr gap: {log_dict['tpr']*100}")
    print(f" - tpr rms: {log_dict['tpr_rms']}")


if __name__ == "__main__":
    main()
