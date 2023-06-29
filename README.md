# GenZ

An instruction fintuned model of Xgen 7B

## Setup

Install dependecies
   ```bash
   pip install -r requirements.txt
   ```

## Tokenizer

The current tokenizer available in huggingface has some issue with adding special tokens like pad_token which is required finetuning. A quick is fix for that is added in the tokenizer file here `utils/tokenizer_xgen.py`

Example usage:
```python
from utils.tokenizer_xgen import XgenTokenizer

tokenizer = XgenTokenizer.from_pretrained('Salesforce/xgen-7b-8k-base', trust_remote_code=True)
tokenizer.pad_token = '<|endoftext|>'
tokenizer.eos_token = '<|endoftext|>'
```

## Finetuning

Coming soon

## Generate

This file allows to do inference from the huggingace model hub and runs a Gradio interface for inference on a specified input. This is an example code which can be modified as needed.

Example usage:

```bash
python generate.py \
    --base_model 'Salesforce/xgen-7b-8k-base'
```

## Benchmark

Coming soon
