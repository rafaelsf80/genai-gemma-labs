""" Basic inference of Gemma 1-2B on GPU using quantized versions (4bit/8bit) and BitsAndBytes
    Requires to accept license (gated repo) on Hugging Face
"""

# Dependencies and restart
#!pip install transformers bitsandbytes accelerate
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# Config for 4bit
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", quantization_config=quantization_config)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))