#  Gemma labs

This repo shows several labs on Gemma 3:

* **Inference Gemma 3-4B-IT** using Hugging Face libraries.
* **Inference Gemma 3-4B-IT and 3n-E4B-IT** using unloth library.
* **Fine-tune Gemma 1-7B** (8.54B parameters), using a `g2-standard-12` machine type with 1xL4 NVidia GPU in **Vertex AI Training**. The model is 4-bit quantized using [NF4](https://arxiv.org/abs/2305.14314) (QLoRA).


## The model: Gemma 3

[Gemma 3](https://www.kaggle.com/models/google/gemma-3) is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. It's released with a [free license for commercial and educational use](https://ai.google.dev/gemma/terms).

Gemma model family are multimodal, decoder-only large language models, multilingual, with open weights, pre-trained variants, and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as a laptop, desktop or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

Gemma 3n is a version of Gemma 3 optimized for devices like phones, laptops, and tablets. This model includes innovations in parameter-efficient processing, including **Per-Layer Embedding (PLE) parameter caching** and a **MatFormer model architecture** that provides the flexibility to reduce compute and memory requirements. These models feature audio input handling, as well as text and visual data. These models were trained with data in **over 140 languages**.

Gemma 3n models use selective parameter activation technology to reduce resource requirements. This technique allows the models to operate at an effective size of 2B and 4B parameters, which is lower than the total number of parameters they contain. For more information on

Available sizes:

* Gemma 3-270M (text only)	
* Gemma 3-1B (text only)	
* Gemma 3-4B	
* Gemma 3-12B	
* Gemma 3-27B
* gemma 3n-E2B (5B parameters)
* gemma 3n-E4B (8B parameters)

Model card for Gemma 3 [here](https://ai.google.dev/gemma/docs/core/model_card_3) and paper [here](https://arxiv.org/abs/2503.19786). 

Model card for Gemma 3n [here](https://ai.google.dev/gemma/docs/gemma-3n/model_card).


## Fine-tuning Gemma 1-7B on Vertex AI Training with QLoRA

The dataset [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) is a dataset of all the quotes retrieved from goodreads quotes. This dataset can be used for multi-label text classification and text generation. The content of each quote is in English and concerns the domain of datasets for NLP and beyond.

Gemma [prompt format](https://ai.google.dev/gemma/docs/formatting) for instruction-tuned models (Gemma 2B-IT and Gemma 7B-IT):
```yaml
<start_of_turn>user
knock knock<end_of_turn>
<start_of_turn>model
who is there<end_of_turn>
<start_of_turn>user
Gemma<end_of_turn>
<start_of_turn>model
Gemma who?<end_of_turn>
``` 

Commands for QLoRA tuning in Vertex AI Training:
```sh
gcloud builds submit --tag europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/gemma-qlora
python3 custom_training.py 
```

> PENDING: Model to be deployed on Vertex AI Prediction.


## References

* Codelab: [Showcasing Agile Safety Classifiers with Gemma](https://codelabs.developers.google.com/codelabs/responsible-ai/agile-classifiers)     
* Codelab: [Using LIT to Analyze Gemma Models in Keras](https://codelabs.developers.google.com/codelabs/responsible-ai/lit-gemma)