""" PEFT tuning of Gemma 7B on GPU using 4bit quantized version and BitsAndBytes
    Tuning takes around 5 min with a T4 GPU in Colab
    Requires to accept license (gated repo) on Hugging Face
"""

# Dependencies and restart
# !pip install accelerate==0.30.1 bitsandbytes==0.43.1 datasets==2.17.0 peft==0.11.1 transformers==4.41.1 trl==0.8.6 -U
import sys
if "google.colab" in sys.modules:
    import IPython
    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)

# Credentials. 
import os
if "google.colab" in sys.modules:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
else:
    os.environ["HF_TOKEN"] = "<REPLACE_WITH_YOUR_TOKEN>"      

# Import packages
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer

model_id = "google/gemma-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])

text = "Quote: Imagination is more"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

os.environ["WANDB_DISABLED"] = "true"

# LoRA config
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# Load dataset
# Abirate/english_quotes are quotes retrieved from "goodreads" quotes. This dataset can be used for multi-label text classification and text generation
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)


# Fine-tuning
import transformers
from trl import SFTTrainer

def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}"
    return [text]

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
trainer.train()

# Inference
text = "Quote: Imagination is"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
