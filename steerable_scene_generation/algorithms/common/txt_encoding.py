from typing import List, Tuple

import torch
import torch.nn as nn

from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer, BatchEncoding, CLIPTextModel


class TxtTokenizer(nn.Module):
    """
    A text tokenizer.
    Supports any model compatible with Hugging Face's AutoTokenizer.
    """

    def __init__(self, version: str, max_length: int):
        super().__init__()

        self.max_length = max_length

        # Load tokenizer and enable fast tokenization if supported.
        self.tokenizer = AutoTokenizer.from_pretrained(version, use_fast=True)

    def forward(self, text: List[str]) -> BatchEncoding:
        """
        Tokenize input text.

          Args:
              text (List[str]): A list of input text strings.

          Returns:
              BatchEncoding: The tokenized text in tensor format.
        """
        # Tokenize text with parallel processing and convert to tensors.
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return batch_encoding

    def get_num_required_tokens(self, text: str) -> int:
        """
        Get the number of required tokens for the input text.
        """
        assert isinstance(text, str)

        # Tokenize without truncation to get full length.
        encoding = self.tokenizer(text, truncation=False)

        return len(encoding["input_ids"])


class TxtTokenEncoder(nn.Module):
    """
    A text token encoder that embeds text tokens into a fixed-size vector.
    Supports any model compatible with Hugging Face's AutoModel.
    """

    def __init__(self, version: str, **hf_kwargs):
        super().__init__()

        # Load the model.
        if "clip" in version:
            self.hf_module = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.hf_module = AutoModel.from_pretrained(version, **hf_kwargs)
        if "t5" in version:
            self.hf_module = self.hf_module.encoder

        # Set output key based on the model type.
        if "pooler_output" in self.hf_module.config.__dict__:
            self.output_key = "pooler_output"
        else:
            self.output_key = "last_hidden_state"

        # Freeze the model to prevent gradient computation.
        self.hf_module = self.hf_module.eval().requires_grad_(False)

        self.hidden_size = self.hf_module.config.hidden_size

    def forward(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """
        Embed input text into a fixed-size vector.

        Args:
            batch_encoding (BatchEncoding): The tokenized text in tensor format.

        Returns:
            torch.Tensor: Encoded text tensor.
        """
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=batch_encoding["attention_mask"].to(self.hf_module.device),
            output_hidden_states=False,
        )

        return outputs[self.output_key]


class TxtEmbedder(nn.Module):
    """
    A text encoder that embeds text into a fixed-size vector.
    Supports any model compatible with Hugging Face's AutoTokenizer and AutoModel.
    """

    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()

        self.tokenizer = TxtTokenizer(version=version, max_length=max_length)
        self.encoder = TxtTokenEncoder(version=version, **hf_kwargs)

        self.hidden_size = self.encoder.hidden_size

    def forward(self, text: List[str]) -> torch.Tensor:
        """
        Embed input text into a fixed-size vector.

        Args:
            text (List[str]): A list of input text strings.

        Returns:
            torch.Tensor: Encoded text tensor.
        """
        batch_encoding = self.tokenizer(text)
        encoding = self.encoder(batch_encoding)
        return encoding


def concat_batch_encodings(batch_encodings: List[BatchEncoding]) -> BatchEncoding:
    """
    Concatenate a list of BatchEncoding objects.

    Args:
        batch_encodings (List[BatchEncoding]): A list of BatchEncoding objects.

    Returns:
        BatchEncoding: The concatenated BatchEncoding object.
    """
    # Concatenate input_ids.
    input_ids = torch.cat([be["input_ids"] for be in batch_encodings], dim=0)

    # Concatenate attention_mask.
    attention_mask = torch.cat([be["attention_mask"] for be in batch_encodings], dim=0)

    return BatchEncoding(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )


def load_t5(
    size: str = "xxl", max_length: int = 256, component: str = "all"
) -> TxtEmbedder | TxtTokenizer | TxtTokenEncoder:
    """
    Args:
        size (str): Size of the T5 model. Should be one of "small", "base", "large",
            "xl", "xxl".
        max_length (int): Maximum sequence length. This is the max number of tokens that
            the input text can have.
        component (str): The component to load. Should be one of "all", "tokenizer",
            "encoder".

    Returns:
        TxtEmbedder | TxtTokenizer | TxtTokenEncoder: The text embedder, tokenizer, or
            encoder, depending on the component.
    """
    if size == "small":
        version = "google/t5-v1_1-small"
    elif size == "base":
        version = "google/t5-v1_1-base"
    elif size == "large":
        version = "google/t5-v1_1-large"
    elif size == "xl":
        version = "google/t5-v1_1-xl"
    elif size == "xxl":
        version = "google/t5-v1_1-xxl"
    else:
        raise ValueError(f"Unsupported size: {size}")

    if component == "tokenizer":
        return TxtTokenizer(version, max_length=max_length)
    elif component == "encoder":
        return TxtTokenEncoder(version, torch_dtype=torch.bfloat16)
    elif component == "all":
        return TxtEmbedder(version, max_length=max_length, torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported component: {component}")


def load_clip(component: str = "all") -> TxtEmbedder | TxtTokenizer | TxtTokenEncoder:
    """
    Args:
        component (str): The component to load. Should be one of "all", "tokenizer",
            "encoder".

    Returns:
        TxtEmbedder | TxtTokenizer | TxtTokenEncoder: The text embedder, tokenizer, or
            encoder, depending on the component.
    """
    version = "openai/clip-vit-large-patch14"
    max_length = 77
    if component == "tokenizer":
        return TxtTokenizer(version, max_length=max_length)
    elif component == "encoder":
        return TxtTokenEncoder(version, torch_dtype=torch.bfloat16)
    elif component == "all":
        return TxtEmbedder(version, max_length=max_length, torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported component: {component}")


def load_bert(
    size: str = "base", max_length: int = 256, component: str = "all"
) -> TxtEmbedder | TxtTokenizer | TxtTokenEncoder:
    """
    Args:
        size (str): Size of the BERT model. Should be one of "tiny", "base", "large".
        max_length (int): Maximum sequence length. This is the max number of tokens that
            the input text can have.
        component (str): The component to load. Should be one of "all", "tokenizer",
            "encoder".

    Returns:
        TxtEmbedder | TxtTokenizer | TxtTokenEncoder: The text embedder, tokenizer, or
            encoder, depending on the component.
    """
    if size == "tiny":
        version = "prajjwal1/bert-tiny"
    elif size == "base":
        version = "google-bert/bert-base-uncased"
    elif size == "large":
        version = "google-bert/bert-large-uncased"
    else:
        raise ValueError(f"Unsupported size: {size}")

    if component == "tokenizer":
        return TxtTokenizer(version, max_length=max_length)
    elif component == "encoder":
        return TxtTokenEncoder(version, torch_dtype=torch.bfloat16)
    elif component == "all":
        return TxtEmbedder(version, max_length=max_length, torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported component: {component}")


def load_txt_encoder_from_config(
    cfg: DictConfig, is_coarse: bool = False, component: str = "all"
) -> Tuple[TxtEmbedder | TxtTokenizer | TxtTokenEncoder, int]:
    """
    Args:
        cfg (DictConfig): The config dictionary. Must have the
            `classifier_free_guidance.txt_encoder` and
            `classifier_free_guidance.txt_encoder_size` keys.
        is_coarse (bool): Whether to load a text encoder for coarse text.
        component (str): The component to load. Should be one of "all", "tokenizer",
            "encoder".

    Returns:
        Tuple[TxtEmbedder | TxtTokenizer | TxtTokenEncoder, int]: The text embedder,
            tokenizer, or encoder, depending on the component and the dimension of the
            output vector.
    """
    encoder = (
        cfg.classifier_free_guidance.txt_encoder_coarse
        if is_coarse
        else cfg.classifier_free_guidance.txt_encoder
    )
    size = (
        cfg.classifier_free_guidance.txt_encoder_coarse_size
        if is_coarse
        else cfg.classifier_free_guidance.txt_encoder_size
    )
    max_length = cfg.classifier_free_guidance.max_length
    if encoder == "t5":
        txt_encoder = load_t5(size=size, max_length=max_length, component=component)
    elif encoder == "clip":
        txt_encoder = load_clip(component=component)
    elif encoder == "bert":
        txt_encoder = load_bert(size=size, max_length=max_length, component=component)
    else:
        raise ValueError("Unsupported txt_encoder:", encoder)
    dim = txt_encoder.hidden_size if component != "tokenizer" else 0
    return txt_encoder, dim
