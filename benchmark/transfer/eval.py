import sys
import os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer

# Set up logger
logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)

dir_path = os.path.dirname(os.path.realpath(__file__))

# Set PATHs
PATH_TO_SENTEVAL = dir_path + "/"
PATH_TO_DATA = dir_path + "/data"

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import torch.nn as nn


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Transformers' model name or path",
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "test", "fasttest"],
        default="fasttest",
        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results",
    )
    parser.add_argument(
        "--task_set",
        type=str,
        choices=["sts", "transfer", "full", "na"],
        default="transfer",
        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'",
    )
    parser.add_argument("--cache_dir", default=None, type=str, help="cache directory")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "MR",
            "CR",
            "MPQA",
            "SUBJ",
            "SST2",
            "TREC",
            "MRPC",
            "SICKRelatedness",
            "STSBenchmark",
        ],
        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden",
    )

    args = parser.parse_args()

    # Load transformers' model checkpoint
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, output_attentions=True, return_dict=True
    )
    model = AutoModel.from_pretrained(
        args.model_name_or_path, config=config, cache_dir=args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    # Set up the tasks
    if args.task_set == "sts":
        args.tasks = [
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "STSBenchmark",
            "SICKRelatedness",
        ]
    elif args.task_set == "transfer":
        args.tasks = ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
    elif args.task_set == "full":
        args.tasks = [
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "STSBenchmark",
            "SICKRelatedness",
        ]
        args.tasks += ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]

    # Set params for SentEval
    if args.mode == "dev" or args.mode == "fasttest":
        # Fast mode
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
        params["classifier"] = {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 128,
            "tenacity": 3,
            "epoch_size": 2,
        }
    elif args.mode == "test":
        # Full mode
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 10}
        params["classifier"] = {
            "nhid": 0,
            "optim": "adam",
            "batch_size": 64,
            "tenacity": 5,
            "epoch_size": 4,
        }
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode("utf-8") for word in s] for s in batch]

        sentences = [" ".join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=True,
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors="pt",
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)

        # Get raw embeddings
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]

        return last_hidden[:, 0].cpu()

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == "dev":
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ["STSBenchmark", "SICKRelatedness"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["dev"]["spearman"][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["devacc"]))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == "test" or args.mode == "fasttest":
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in [
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "STSBenchmark",
            "SICKRelatedness",
        ]:
            task_names.append(task)
            if task in results:
                if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                    scores.append(
                        "%.2f" % (results[task]["all"]["spearman"]["mean"] * 100)
                    )
                else:
                    scores.append("%.2f" % (results[task]["spearman"] * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["acc"]))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    main()
