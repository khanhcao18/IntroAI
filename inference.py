"""Load the original ABSA models and return one row per detected aspect."""

import logging
import os
import pickle
from functools import cache, partial
from pathlib import Path

import torch

from UI import infer_joint_model, infer_lstm, infer_lstm_attention
from UI.infer_ml import predict_input


ROOT = Path(__file__).resolve().parent
TOKENIZER_ID = "bkai-foundation-models/vietnamese-bi-encoder"
TOKENIZER_REVISION = "84f9d9a"
SMOKE_REVIEW = "Camera chụp ảnh đẹp nhưng pin hết khá nhanh."
LOGGER = logging.getLogger(__name__)

SVM_ASPECTS = (
    "BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL",
    "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE",
)
SVM_SENTIMENTS = {0: "Neutral", 1: "Negative", 2: "Positive"}


class InvalidReview(ValueError):
    """The user can fix the review and submit it again."""


class TokenizerUnavailable(RuntimeError):
    """The tokenizer could not be loaded."""


def prepare_review(review):
    if not review or not review.strip():
        raise InvalidReview("Vui lòng nhập một nhận xét về điện thoại.")

    from Data.preprocess_for_input_infer import preprocess_text

    prepared = preprocess_text(review)
    if not prepared.strip():
        raise InvalidReview("Nhận xét cần có nội dung văn bản.")
    return prepared


def predict_review(review, predictor):
    return predictor(prepare_review(review))


def decode(categories, sentiments, aspects, sentiment_labels):
    rows = []
    for index, (detected, sentiment) in enumerate(zip(categories, sentiments, strict=True)):
        if detected == 1:
            label = None if sentiment == -1 else sentiment_labels[int(sentiment)]
            rows.append((aspects[index], label))
    return rows


def load_svm(model_path, vectorizer_path):
    # These pickles are the fitted artifacts shipped with the project.
    with model_path.open("rb") as source:
        model = pickle.load(source)
    with vectorizer_path.open("rb") as source:
        vectorizer = pickle.load(source)

    def predict(prepared):
        output = predict_input(prepared, model, vectorizer)
        category_count = len(SVM_ASPECTS)
        if output.shape != (1, 2 * category_count):
            raise ValueError(f"Unexpected SVM output shape: {output.shape}")
        return decode(output[0, :category_count], output[0, category_count:], SVM_ASPECTS, SVM_SENTIMENTS)

    return predict


@cache
def tokenizer_source(local_path):
    from transformers import AutoTokenizer

    try:
        if local_path:
            return AutoTokenizer.from_pretrained(local_path, use_fast=False, local_files_only=True)
        return AutoTokenizer.from_pretrained(
            TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=False,
        )
    except OSError as error:
        # Cache the failed startup attempt as well, so all three neural models
        # can report the same cause after a single download attempt.
        LOGGER.warning("Could not load the BKAI tokenizer: %s", error)
        return None


def load_neural(checkpoint, *, architecture, labels, use_attention_mask=True):
    tokenizer = tokenizer_source(os.getenv("ABSA_TOKENIZER_PATH"))
    if tokenizer is None:
        raise TokenizerUnavailable("Thiếu bộ tách từ BKAI. Xem hướng dẫn cài đặt trong README.")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if use_attention_mask:
        embeddings = state["word_embedder.weight"]
        model = architecture(
            torch.nn.Embedding(*embeddings.shape), labels.aspect2idx, labels.sentiment2idx,
        )
    else:
        embeddings = state["embed.weight"]
        model = architecture(torch.empty_like(embeddings))

    embedding_rows = len(embeddings)
    if embedding_rows not in {tokenizer.vocab_size, len(tokenizer)}:
        raise ValueError("Tokenizer vocabulary size differs from the checkpoint.")
    model.load_state_dict(state, strict=True)
    model.eval()

    def predict(prepared):
        encoding = tokenizer(
            prepared, padding="max_length", truncation=True, max_length=128, return_tensors="pt",
        )
        tokens = encoding["input_ids"]
        if int(tokens.min()) < 0 or int(tokens.max()) >= embedding_rows:
            raise InvalidReview("Nhận xét chứa mã từ nằm ngoài bộ từ vựng của mô hình.")
        with torch.inference_mode():
            if use_attention_mask:
                output = model(tokens, labels=None, mask=encoding["attention_mask"])
            else:
                output = model(tokens)
        return decode(
            output["pred_category"][0], output["pred_sentiment"][0],
            labels.idx2aspect, labels.idx2sentiment,
        )

    return predict


MODEL_LOADERS = {
    "SVM": (("Model/ML_Model", "Model/vectorizer_for_ML"), load_svm),
    "LSTM": (
        ("UI/CAE_checkpoint50.pth",),
        partial(load_neural, architecture=infer_lstm.LSTMBase, labels=infer_lstm),
    ),
    "LSTM + Attention": (
        ("UI/ATAE_checkpoint20.pth",),
        partial(
            load_neural, architecture=infer_lstm_attention.ATAE_LSTM,
            labels=infer_lstm_attention, use_attention_mask=False,
        ),
    ),
    "Joint Model Shared": (
        ("UI/CAE_checkpoint41.pth",),
        partial(load_neural, architecture=infer_joint_model.JointModel, labels=infer_joint_model),
    ),
}


def load_models(root=ROOT):
    """Enable a model only after its files load and raw-text inference succeeds."""
    predictors = {}
    unavailable = {}

    for name, (filenames, load) in MODEL_LOADERS.items():
        missing = [filename for filename in filenames if not (root / filename).is_file()]
        if missing:
            unavailable[name] = "Thiếu tệp: " + ", ".join(missing)
            continue
        try:
            predictor = load(*(root / filename for filename in filenames))
            predict_review(SMOKE_REVIEW, predictor)
        except TokenizerUnavailable as error:
            unavailable[name] = str(error)
        except ModuleNotFoundError as error:
            LOGGER.exception("Missing dependency for %s", name)
            unavailable[name] = f"Thiếu thư viện `{error.name}`."
        except Exception:
            LOGGER.exception("Could not initialize %s", name)
            unavailable[name] = "Không thể tải hoặc chạy mô hình. Xem nhật ký máy chủ."
        else:
            predictors[name] = predictor

    unavailable["Joint Model without Sharing"] = "Thiếu kiến trúc và trọng số tương ứng."
    return predictors, unavailable
