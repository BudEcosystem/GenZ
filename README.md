<p align="center" width="100%">
<a ><img src="https://github.com/BudEcosystem/GenZ/blob/main/assets/genz.png" alt="WizardLM" style="width: 20%; min-width: 300px; display: block; margin: auto;"></a>
</p>
The most capable commercially usable Instruct Finetuned LLM yet with 8K input token length, latest information & better coding. 

[Genz 7B](https://huggingface.co/budecosystem/genz-7b) | [Genz 13B](https://huggingface.co/budecosystem/genz-13b) | [Genz 13B v2](https://huggingface.co/budecosystem/genz-13b-v2)

## Announcement

- [26 Jul 2023] We have released Genz 13B v2 model with better eval than v1. Download the model from huggingface( [Genz 13B v2](https://huggingface.co/budecosystem/genz-13b-v2) )
- [20 Jul 2023] We have released Genz 13B model. Download the model from huggingface([Genz 13B](https://huggingface.co/budecosystem/genz-13b)).


## Setup

Install dependencies
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

This file allows to do inference from the huggingface model hub and runs a Gradio interface for inference on a specified input. This is an example code that can be modified as needed.

Example usage:

```bash
python generate.py \
    --base_model 'budecosystem/genz-13b-v2'
```

## Benchmark

| Model Name | MT Bench | Vicuna Bench | MMLU | Human Eval | Hellaswag | BBH |
|------------|----------|--------------|------|------------|-----------|-----|
| [Genz 13B](https://huggingface.co/budecosystem/genz-13b) | 6.12 | 86.1 | 53.62 | 17.68 | 77.38 | 37.76 |
| [Genz 13B v2](https://huggingface.co/budecosystem/genz-13b-v2) | 6.79 | 87.2 | 53.68 | 21.95 | 77.48 | 38.1 |

&nbsp;<br>

MT Bench score

<img src="https://github.com/BudEcosystem/GenZ/blob/main/assets/mt_bench_score.png" width="500">
