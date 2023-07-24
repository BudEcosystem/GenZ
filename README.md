<p align="center" width="100%">
<a ><img src="https://github.com/BudEcosystem/GenZ/blob/main/assets/genz.png" alt="WizardLM" style="width: 20%; min-width: 300px; display: block; margin: auto;"></a>
</p>
The most capable commercially usable Instruct Finetuned LLM yet with 8K input token length, latest information & better coding. 

[Genz 7B](https://huggingface.co/budecosystem/genz-7b) | [Genz 13B](https://huggingface.co/budecosystem/genz-13b)

## Announcement

- [20 Jul 2023] We have released Genz13B model. Download the model from huggingface([Genz13B](https://huggingface.co/budecosystem/genz-13b)). 

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

| Model Name | Vicuna Bench | MMLU | Human Eval | Hellaswag |
|-------------|-------------|------|------------|-----------|
| [Genz 13B](https://huggingface.co/budecosystem/genz-13b) | 86.2 | 53.62 | 17.68 | |

&nbsp;<br>

MT Bench score

<img src="https://github.com/BudEcosystem/GenZ/blob/main/assets/mt_bench_score.png" width="500">
