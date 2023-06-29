
import os
import sys

import fire
import gradio as gr
import torch

from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.prompt import get_prompt

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
):
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if lora_weights:
        print("Loading lora layer")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model.half()
    model.eval()

    def evaluate(
        instruction,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = get_prompt(instruction)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield output

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Query",
                placeholder="Ask me anything",
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=8192, step=1, value=128, label="Max tokens"
            )
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="GenZ",
        description="An instruction fintuned model of Xgen 7B",
    ).queue().launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    fire.Fire(main)
