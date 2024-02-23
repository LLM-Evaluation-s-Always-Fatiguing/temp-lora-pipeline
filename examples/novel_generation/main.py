from typing import List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import TaskType, LoraConfig
from temp_lora_pipeline import TempLoraTuningConfig, TempLoraCompletionPipeline


def find_all_linear_names(model: nn.Module) -> List[str]:
    linear_class = nn.Linear
    lora_module_names = set([])

    for name, module in model.named_modules():
        if isinstance(module, linear_class):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


model_name_or_path = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
streamer = TextStreamer(tokenizer)

with open("349.txt", "r", encoding="utf-8") as f:
    book = ""
    for line in f.readlines():
        book += line
    book.strip()

pipeline = TempLoraCompletionPipeline(
    model=model,
    tokenizer=tokenizer,
    lora_config=LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model=model)
    ),
    lora_training_config=TempLoraTuningConfig(num_epoch=2)
)
outputs = pipeline(
    text_inputs="Below is a novel.\n" + book + "\n Continue the book based on its content.",
    max_new_tokens=100000,
    min_new_tokens=100000,
    do_sample=True,
    num_beams=1,
    num_return_sequences=1,
    streamer=streamer,
    return_full_text=False,
)
