""" Basic inference of Gemma 1-2B on CPU using Hugging face libraries
    Requires to accept license (gated repo) on Hugging Face
"""

# Dependencies and restart
#!pip install -q -U accelerate
import sys
if "google.colab" in sys.modules:
    import IPython
    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
    
# Credentials. If not using Colab, replace userdata.get (Colab API)
import os
from google.colab import userdata # only for Colab
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')

# Import packages
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
print(model)
# For GPU and bfloat16 type
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", torch_dtype=torch.bfloat16)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")
# For GPU:
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))





