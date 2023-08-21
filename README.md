---

<div align="center"><h1 align="center">~ GenZ ~</h1><img src="https://github.com/BudEcosystem/GenZ/blob/main/assets/genz-logo.png" width=150></div>


<p align="center"><i>Democratizing access to LLMs for the open-source community.<br>Let's advance AI, together. </i></p>

---


## Introduction 🎉

Welcome to **GenZ**, an advanced Large Language Model (LLM) fine-tuned on the foundation of Meta's open-source Llama V2 13B parameter model. At Bud Ecosystem, we believe in the power of open-source collaboration to drive the advancement of technology at an accelerated pace. Our vision is to democratize access to fine-tuned LLMs, and to that end, we will be releasing a series of models across different parameter counts (7B, 13B, and 70B) and quantizations (32-bit and 4-bit) for the open-source community to use, enhance, and build upon. 

<p align="center"><img src="https://github.com/BudEcosystem/GenZ/blob/main/assets/MTBench_CompareChart_28July2023.png" width="500"></p>

The smaller quantization version of our models makes them more accessible, enabling their use even on personal computers. This opens up a world of possibilities for developers, researchers, and enthusiasts to experiment with these models and contribute to the collective advancement of language model technology. 

GenZ isn't just a powerful text generator—it's a sophisticated AI assistant, capable of understanding and responding to user prompts with high-quality responses. We've taken the robust capabilities of Llama V2 and fine-tuned them to offer a more user-focused experience. Whether you're seeking informative responses or engaging interactions, GenZ is designed to deliver.

And this isn't the end. It's just the beginning of a journey towards creating more advanced, more efficient, and more accessible language models. We invite you to join us on this exciting journey. 🚀

---

<h2 align="center">Milestone Releases ️🏁</h2>

**[21 August 2023]**
[_GenZ-70B_](https://huggingface.co/budecosystem/genz-70b) : We marked an important milestone with the release of the Genz 70B model. The journey began here, and you can partake in it by downloading the model from [Hugging Face](https://huggingface.co/budecosystem/genz-70b).

**[4 August 2023]**
[_GenZ Vision Assistant 7B_](https://huggingface.co/budecosystem/genz-mm-vt-7b) : Announcing our multimodal Genz Vision Assistant 7B. This is a multimodal AI model fine-tuned to understand text and visual inputs to provide contextually relevant responses. Download the model from [HuggingFace](https://huggingface.co/budecosystem/genz-mm-vt-7b).

**[27 July 2023]**
[_GenZ-13B V2 (ggml)_](https://huggingface.co/budecosystem/genz-13b-v2-ggml) : Announcing our GenZ-13B v2 with ggml. This variant of GenZ can run inferencing using only CPU and without the need of GPU. Download the model from [HuggingFace](https://huggingface.co/budecosystem/genz-13b-v2-ggml).

**[27 July 2023]**
[_GenZ-13B V2 (4-bit)_](https://huggingface.co/budecosystem/genz-13b-v2-4bit) : Announcing our GenZ-13B v2 with 4-bit quantisation. Enabling inferencing with much lesser GPU memory than the 32-bit variant. Download the model from [HuggingFace](https://huggingface.co/budecosystem/genz-13b-v2-4bit).

**[26 July 2023]**
[_GenZ-13B V2_](https://huggingface.co/budecosystem/genz-13b-v2) : We're excited to announce the release of our Genz 13B v2 model, a step forward with improved evaluation results compared to v1. Experience the advancements by downloading the model from [HuggingFace](https://huggingface.co/budecosystem/genz-13b-v2).

**[20 July 2023]**
[_GenZ-13B_](https://huggingface.co/budecosystem/genz-13b) : We marked an important milestone with the release of the Genz 13B model. The journey began here, and you can partake in it by downloading the model from [Hugging Face](https://huggingface.co/budecosystem/genz-13b).

---


<img src="https://github.com/BudEcosystem/GenZ/blob/main/assets/screenshot_genz13bv2.png" width="100%">

| ![Python](https://github.com/BudEcosystem/GenZ/blob/main/assets/Python.gif) | ![Poem](https://github.com/BudEcosystem/GenZ/blob/main/assets/Poem.gif) | ![Email](https://github.com/BudEcosystem/GenZ/blob/main/assets/Email.gif)
|:--:|:--:|:--:| 
| *Code Generation* | *Poem Generation* | *Email Generation* |

<!--
<p align="center"><img src="https://github.com/adrot-dev/git-test/blob/main/assets/Python.gif" width="33%" alt="Python Code"><img src="https://github.com/adrot-dev/git-test/blob/main/assets/Poem.gif" width="33%"><img src="https://github.com/adrot-dev/git-test/blob/main/assets/Email.gif" width="33%"></p>
-->

---

<h2 align="center">Getting Started on GitHub 💻</h2>

Ready to dive in? Here's how you can get started with our models on GitHub.

<h3 align="center">1️⃣ : Clone our GitHub repository</h3>


First things first, you'll need to clone our repository. Open up your terminal, navigate to the directory where you want the repository to be cloned, and run the following command:

```bash
git clone https://github.com/BudEcosystem/GenZ.git
```

<h3 align="center">2️⃣ : Install dependencies</h3>

Navigate into the cloned repository and install the necessary dependencies with the following command:

```bash
pip install -r requirements.txt
```

<h3 align="center">3️⃣ : Generate responses</h3>

Now that your model is fine-tuned, you're ready to generate responses. You can do this using our generate.py script, which runs inference from the Hugging Face model hub and presents a Gradio interface for inference on a specified input. Here's an example usage:

```bash
python generate.py --base_model 'budecosystem/genz-13b-v2'
```

>☝🏻We have made it convenient to do inference with our models by packaging the code into an easy to use Gradio interface.

The prompt template is integrated into the prompt, as a prefix. The template:

```
A chat between a curious user and an artificial intelligence assistant.\ 
The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: Hi, how are you?
ASSISTANT: 
```

✅ Now that you are up to speed on how to get the model up and running for inference, let’s take a look at how you can customize the model by way of fine-tuning the model with datasets relevant to your specific usecase.

---

<h2 align="center">Fine-tuning 🎯</h2>


It's time to upgrade the model by fine-tuning the model. You can do this using our provided finetune.py script. Here's an example command:

```bash
python finetune.py \
   --model_name Salesforce/xgen-7b-8k-base \
   --data_path dataset.json \
   --output_dir output \
   --trust_remote_code \
   --prompt_column instruction \
   --response_column output \
   --pad_token_id 50256
```

---

<h2 align="center">Getting Started on Hugging Face 🤗</h2>

Getting up and running with our models on Hugging Face is a breeze. Follow these steps:

<h3 align="center">1️⃣ : Import necessary modules</h3>


Start by importing the necessary modules from the ‘transformers’ library and ‘torch’.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
```

<h3 align="center">2️⃣ : Load the tokenizer and the model</h3>

Next, load up the tokenizer and the model for ‘budecosystem/genz-13b’ from Hugging Face using the ‘from_pretrained’ method.

```python
tokenizer = AutoTokenizer.from_pretrained("budecosystem/genz-13b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("budecosystem/genz-13b", torch_dtype=torch.bfloat16)
```

<h3 align="center">3️⃣ : Generate responses</h3>


Now that you have the model and tokenizer, you're ready to generate responses. Here's how you can do it:

```python
inputs = tokenizer("The meaning of life is", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0]))
```

In this example, "The meaning of life is" is the prompt template used for inference. You can replace it with any string you like.

Want to interact with the model in a more intuitive way? We have a Gradio interface set up for that. Head over to our GitHub page, clone the repository, and run the ‘generate.py’ script to try it out. Happy experimenting! 😄

<h2 align="center">Fine-tuning 🎯</h2>


It's time to upgrade the model by fine-tuning the model. You can do this using our provided finetune.py script. Here's an example command:

```bash
python finetune.py \
   --model_name Salesforce/xgen-7b-8k-base \
   --data_path dataset.json \
   --output_dir output \
   --trust_remote_code \
   --prompt_column instruction \
   --response_column output \
   --pad_token_id 50256
```

---

<h2 align="center">Bonus: Colab Notebooks 📚 <b><i>(WIP)</i></b></h2>


Looking for an even simpler way to get started with GenZ? We've got you covered. We've prepared a pair of detailed Colab notebooks - one for Inference and one for Fine-tuning. These notebooks come pre-filled with all the information and code you'll need. All you'll have to do is run them!

Keep an eye out for these notebooks. They'll be added to the repository soon!

---

<h2 align="center">Why Use GenZ? 💡</h2>


You might be wondering, "Why should I choose GenZ over a pretrained model?" The answer lies in the extra mile we've gone to fine-tune our models.

While pretrained models are undeniably powerful, GenZ brings something extra to the table. We've fine-tuned it with curated datasets, which means it has additional skills and capabilities beyond what a pretrained model can offer. Whether you need it for a simple task or a complex project, GenZ is up for the challenge.

What's more, we are committed to continuously enhancing GenZ. We believe in the power of constant learning and improvement. That's why we'll be regularly fine-tuning our models with various curated datasets to make them even better. Our goal is to reach the state of the art and beyond - and we're committed to staying the course until we get there.

But don't just take our word for it. We've provided detailed evaluations and performance details in a later section, so you can see the difference for yourself.

Choose GenZ and join us on this journey. Together, we can push the boundaries of what's possible with large language models.

---

<h2 align="center">Model Card for GenZ 13B 📄</h2>

Here's a quick overview of everything you need to know about GenZ 13B.

<h3 align="center">Model Details:</h3>


- Developed by: Bud Ecosystem
- Base pretrained model type: Llama V2 13B
- Model Architecture: GenZ 13B, fine-tuned on Llama V2 13B, is an auto-regressive language model that employs an optimized transformer architecture. The fine-tuning process for GenZ 13B leveraged Supervised Fine-Tuning (SFT)
- License: The model is available for commercial use under a custom commercial license. For more information, please visit: [Meta AI Model and Library Downloads](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

---

<h2 align="center">Intended Use 💼</h2>

When we created GenZ 13B, we had a clear vision of how it could be used to push the boundaries of what's possible with large language models. We also understand the importance of using such models responsibly. Here's a brief overview of the intended and out-of-scope uses for GenZ 13B.

<h3 align="center">Direct Use</h3>

GenZ 13B is designed to be a powerful tool for research on large language models. It's also an excellent foundation for further specialization and fine-tuning for specific use cases, such as:
- Text summarization
- Text generation
- Chatbot creation
- And much more!

<h3 align="center">Out-of-Scope Use 🚩</h3>

While GenZ 13B is versatile, there are certain uses that are out of scope:

- Production use without adequate assessment of risks and mitigation
- Any use cases which may be considered irresponsible or harmful
- Use in any manner that violates applicable laws or regulations, including trade compliance laws
- Use in any other way that is prohibited by the Acceptable Use Policy and Licensing Agreement for Llama 2

Remember, GenZ 13B, like any large language model, is trained on a large-scale corpora representative of the web, and therefore, may carry the stereotypes and biases commonly encountered online.

<h3 align="center">Recommendations 🧠</h3>

We recommend users of GenZ 13B to consider fine-tuning it for the specific set of tasks of interest. Appropriate precautions and guardrails should be taken for any production use. Using GenZ 13B responsibly is key to unlocking its full potential while maintaining a safe and respectful environment.

---

<h2 align="center">Training Details 📚</h2>

When fine-tuning GenZ 13B, we took a meticulous approach to ensure we were building on the solid base of the pretrained Llama V2 13B model in the most effective way. Here's a look at the key details of our training process:

<h3 align="center">Fine-Tuning Training Data</h3>

For the fine-tuning process, we used a carefully curated mix of datasets. These included data from OpenAssistant, an instruction fine-tuning dataset, and Thought Source for the Chain Of Thought (CoT) approach. This diverse mix of data sources helped us enhance the model's capabilities across a range of tasks.

<h3 align="center">Fine-Tuning Procedure</h3>

We performed a full-parameter fine-tuning using Supervised Fine-Tuning (SFT). This was carried out on 4 A100 80GB GPUs, and the process took under 100 hours. To make the process more efficient, we used DeepSpeed's ZeRO-3 optimization.

<h3 align="center">Tokenizer</h3>

We used the SentencePiece tokenizer during the fine-tuning process. This tokenizer is known for its capability to handle open-vocabulary language tasks efficiently.


<h3 align="center">Hyperparameters</h3>


Here are the hyperparameters we used for fine-tuning:

| Hyperparameter | Value |
| -------------- | ----- |
| Warmup Ratio | 0.04 |
| Learning Rate Scheduler Type | Cosine |
| Learning Rate | 2e-5 |
| Number of Training Epochs | 3 |
| Per Device Training Batch Size | 4 |
| Gradient Accumulation Steps | 4 |
| Precision | FP16 |
| Optimizer | AdamW |

---

<h2 align="center">Evaluations 🎯</h2>

Evaluating our model is a key part of our fine-tuning process. It helps us understand how our model is performing and how it stacks up against other models. Here's a look at some of the key evaluations for GenZ 13B:

<h3 align="center">Benchmark Comparison</h3>

We've compared GenZ V1 with V2 to understand the improvements our fine-tuning has achieved.

| Model Name | MT Bench | Vicuna Bench | MMLU | Human Eval | Hellaswag | BBH |
|:----------:|:--------:|:------------:|:----:|:----------:|:---------:|:----:|
| Genz 13B   | 6.12     | 86.1         | 53.62| 17.68      | 77.38     | 37.76|
| Genz 13B v2| 6.79     | 87.2         | 53.68| 21.95      | 77.48     | 38.1 |

<h3 align="center">MT Bench Score</h3>

A key evaluation metric we use is the MT Bench score. This score provides a comprehensive assessment of our model's performance across a range of tasks.

We're proud to say that our model performs at a level that's close to the Llama-70B-chat model on the MT Bench and top of the list among 13B models.

<p align="center"><img src="https://github.com/BudEcosystem/GenZ/blob/main/assets/mt_bench_score.png" width="500"></p>

In the transition from GenZ V1 to V2, we noticed some fascinating performance shifts. While we saw a slight dip in coding performance, two other areas, Roleplay and Math, saw noticeable improvements.

---

<h2 align="center">Looking Ahead 👀</h2>

We're excited about the journey ahead with GenZ. We're committed to continuously improving and enhancing our models, and we're excited to see what the open-source community will build with them. We believe in the power of collaboration, and we can't wait to see what we can achieve together.

Remember, we're just getting started. This is just the beginning of a journey that we believe will revolutionize the world of large language models. We invite you to join us on this exciting journey. Together, we can push the boundaries of what's possible with AI. 🚀

---
