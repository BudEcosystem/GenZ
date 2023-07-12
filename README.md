# GenZ

The most capable commercially usable Instruct Finetuned LLM yet with 8K input token length, latest information & better coding. 

Check the model in HuggingFace -> [Genz 7B](https://huggingface.co/budecosystem/genz-7b)

## Setup

Install dependecies
   ```bash
   pip install -r requirements.txt
   ```


## Finetuning

```bash
python finetune.py
   --model_name Salesforce/xgen-7b-8k-base
   --data_path dataset.json
   --output_dir output
   --trust_remote_code
   --prompt_column instruction
   --response_column output
   --pad_token_id 50256
```

## Generate

This file allows to do inference from the huggingace model hub and runs a Gradio interface for inference on a specified input. This is an example code which can be modified as needed.

Example usage:

```bash
python generate.py \
    --base_model 'budecosystem/genz-7b'
```

## Benchmark

Coming soon
