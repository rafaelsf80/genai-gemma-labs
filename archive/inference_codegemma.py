""" Basic inference of Codegemma using Keras 3
    IMPORTANT: You must accept Gemma license conditions on Kaggle page. Otherwise, you will get a 403 error when doing `.from_preset()` 
    TEMPORAL ERROR: Keras3 works in Colab but not in Vertex AI Workbench Instances
"""

# Dependencies and restart
#!pip3 install -q -U keras-nlp
#!pip3 install -q -U keras>=3
import sys
if "google.colab" in sys.modules:
    import IPython
    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
    
# Credentials. If not using Colab, replace userdata.get (Colab API) with a string with your credentials
import os
from google.colab import userdata # only for Colab
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

# Import packages
import keras
import keras_nlp

# Backend for Keras 3
os.environ["KERAS_BACKEND"]   = "tensorflow"  # Or "jax" or "torch".

# Run at half precision
keras.config.set_floatx("bfloat16")

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("code_gemma_2b_en")
gemma_lm.summary()

# Fill-in-the-nidlle code completion
BEFORE_CURSOR = "<|fim_prefix|>"
AFTER_CURSOR = "<|fim_suffix|>"
AT_CURSOR = "<|fim_middle|>"
FILE_SEPARATOR = "<|file_separator|>"

END_TOKEN = gemma_lm.preprocessor.tokenizer.end_token
stop_tokens = (BEFORE_CURSOR, AFTER_CURSOR, AT_CURSOR, FILE_SEPARATOR, END_TOKEN)
stop_token_ids = tuple(gemma_lm.preprocessor.tokenizer.token_to_id(x) for x in stop_tokens)

# Helper function
def format_completion_prompt(before, after):
    return f"{BEFORE_CURSOR}{before}{AFTER_CURSOR}{after}{AT_CURSOR}"

before = "import "
after = """if __name__ == "__main__":\n    sys.exit(0)"""
prompt = format_completion_prompt(before, after)
print(prompt)

gemma_lm.generate(prompt, stop_token_ids=stop_token_ids, max_length=128)
# Output: The model provides `sys` as the suggested code completion.

