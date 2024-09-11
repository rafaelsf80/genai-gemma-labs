""" Basic inference of Gemma 2-9B on T4 GPU (Colab)
    Requires to accept license (gated repo) on Hugging Face
    Inference is slow on T4 GPU: pipe() can take up to 5 min
"""

#!pip install -U accelerate peft bitsandbytes transformers trl datasets
 
from transformers import pipeline
import torch

model_id = "google/gemma-2-9b-it"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token="YOUR_HF_TOKEN" # Replace with HF_TOKEN. Gemma 2-9B is a gated repo in HF

)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]


outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=False,
)

assistant_response = outputs[0]["generated_text"][-1]["content"]
print(assistant_response)