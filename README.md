# ABSA

A Vietnamese aspect-based sentiment analysis demo for smartphone reviews. Aspect Category Detection (ACD) identifies the topics discussed in a review. Sentiment Classification (SC) assigns Positive, Negative, or Neutral to each detected topic.

## Install and run

From the repository root, with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. Open the local URL Gradio prints, typically `http://127.0.0.1:7860`. Select a model and enter a Vietnamese review, or click a sample to fill the textbox.

The first run downloads the [BKAI PhoBERT tokenizer files](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder/tree/84f9d9a). The application uses the trained embedding matrices from the checkpoints. It runs on CPU.

### Run offline or on a server

For offline use, export the tokenizer once on a machine with internet access:

```bash
python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("bkai-foundation-models/vietnamese-bi-encoder", revision="84f9d9a", use_fast=False).save_pretrained("./absa-tokenizer")'
ABSA_TOKENIZER_PATH=./absa-tokenizer python app.py
```

Copy the tokenizer directory and the model artifacts below to any offline host. To listen on all interfaces behind your hosting provider's proxy, set Gradio's server address and port:

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python app.py
```

## Model availability

| Model | Artifact | Serving details |
| --- | --- | --- |
| SVM | `Model/ML_Model` and `Model/vectorizer_for_ML` | RBF SVM and the original fitted bag-of-words vectorizer. Both pickles were saved with scikit-learn 1.6.0. |
| LSTM | `UI/CAE_checkpoint50.pth` | Loads strictly into both original `LSTMBase` definitions. The file's training provenance is undocumented. ACD threshold 0.40. |
| LSTM + Attention | `UI/ATAE_checkpoint20.pth` | Uses epoch 20, the checkpoint selected by the original inference script. ACD threshold 0.25. |
| Joint Model Shared | `UI/CAE_checkpoint41.pth` | Joint attention with a shared sentiment classifier. ACD threshold 0.40. |
| Joint Model without Sharing | Unavailable | Requires the original unshared model architecture, matching trained weights, and its label configuration. |

The selector lists models that load successfully and pass a raw Vietnamese review at startup. Check **Mô hình khả dụng** in the app for any startup failure. Use the SVM pickle files supplied by the project.

The demo uses the original review preprocessing and each model's original label order. A dash in the results means the SVM detected an aspect and its sentiment classifier returned the missing-sentiment value. The results table shows model decisions without confidence scores.

## Code layout

- `app.py` builds the Gradio Blocks interface.
- `inference.py` checks artifacts, loads models, and decodes ACD and SC outputs.
- `UI/infer_*.py` provides the original inference architectures and SVM prediction path.
- `Data/preprocess_for_input_infer.py` provides the Vietnamese review preprocessing.
- `Model/` contains the training scripts and fitted SVM artifacts.
