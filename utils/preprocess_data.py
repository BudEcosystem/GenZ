


IGNORE_INDEX = -100

def prepare_data(dataset, data_args, tokenizer):

    template = {
        "prefix": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        "prompt": "USER: {query} ASSISTANT: ",
        "sep": "\n"
    }

    prompt_col = data_args.prompt_column
    response_col = data_args.response_column
    history_col = data_args.history_column

    def get_dialog(examples):
        dialogs = []
        
        for i in range(len(examples[prompt_col])):
            query = template['prefix'] + template['sep'] + template['prompt'].format(query=examples[prompt_col][i])
            response = examples[response_col][i]
            conv = [query, response]

            for turn, (user, bot) in enumerate(examples[history_col][i]):
                conv.append(template['prompt'].format(query=user))
                conv.append(bot)
            dialogs.append(conv)
        
        return dialogs


    def preprocess(item):
        #data format = {"prompt":"", "response": "", "history": []}
        
        dialogs = get_dialog(item)
        model_inputs = {"input_ids": [], "labels": []}
        
        for dialog in dialogs:
            input_ids, labels = [], []

            for i in range(len(dialog)//2):
                source_ids = tokenizer.encode(text=dialog[2*i], add_special_tokens=False)
                target_ids = tokenizer.encode(text=dialog[2*i+1], add_special_tokens=False)
                input_ids += source_ids + [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]
                labels += [IGNORE_INDEX] * (len(source_ids) + 1) + target_ids + [tokenizer.eos_token_id]

            model_inputs["input_ids"].append(input_ids[:data_args.max_length])
            model_inputs["labels"].append(labels[:data_args.max_length])
        
        return model_inputs
    

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example["labels"]],
                             skip_special_tokens=False)
        ))

    dataset = dataset.map(
        preprocess,
        batched=True,
        desc="Running tokenizer on dataset"
    )

    print_supervised_dataset_example(dataset[0])

    dataset = dataset.train_test_split(test_size=data_args.split_ratio)
    trainer_kwargs = {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}

    return trainer_kwargs