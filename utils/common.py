

from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from datasets import load_dataset

from .config import ModelArguments, DataArguments

IGNORE_INDEX = -100

def parse_args():

    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args

def load_data(data_args):

    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.data_path)
    else:
        data = load_dataset(data_args.data_path)
    
    return data[data_args.split]
