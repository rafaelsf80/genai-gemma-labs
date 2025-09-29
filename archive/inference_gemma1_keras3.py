""" Basic inference of Gemma 1-2B using Keras 3
    Inference with CPU takes several minutes
    IMPORTANT: You must accept Gemma license conditions on Kaggle page. Otherwise, you will get a 403 error when doing `.from_preset()` 
    TEMPORAL ERROR: Keras3 works in Colab but not in Vertex AI Workbench Instances
"""

# Dependencies and restart
#!pip install -q -U keras-nlp
#!pip install -q -U keras>=3
import sys
if "google.colab" in sys.modules:
    import IPython
    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
    
# Credentials. If not using Colab, replace userdata.get (Colab API)
import os
from google.colab import userdata # only for Colab
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')


# Import packages
import keras
import keras_nlp

# Backend for Keras 3
os.environ["KERAS_BACKEND"]   = "tensorflow"  # Or "jax" or "torch".

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

gemma_lm.summary()

print(gemma_lm.generate("What is the meaning of life?", max_length=64))

# Second inference should be very fast, thanks to XLA and tensorflow/jax backends
print(gemma_lm.generate("How does the brain work?", max_length=64))

# Batch inference
print(gemma_lm.generate(
    ["What is the meaning of life?",
     "How does the brain work?"],
    max_length=64))

# Optionally try a different sampler
#gemma_lm.compile(sampler="top_k")
#gemma_lm.generate("What is the meaning of life?", max_length=64) #, top_k=3)
