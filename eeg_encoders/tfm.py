from functools import lru_cache

import torch
import torch.nn as nn
from einops import rearrange

from config import config
from pyhealth.datasets import create_sample_dataset
from pyhealth.models import TFMTokenizer
from pyhealth.models.tfm_tokenizer import get_stft_torch


@lru_cache(maxsize=None)
def build_tfm_schema_dataset(data_name):
    data_name = data_name.lower()
    data_config = getattr(config.datasets, data_name.upper())
    signal_length = int(data_config["resampling_rate"] * data_config["signal_len"])
    output_mode = "binary" if data_config["task"] == "binary" else "multiclass"

    samples = []
    for label in range(data_config["classes"]):
        samples.append(
            {
                "patient_id": f"{data_name}_schema_patient_{label}",
                "record_id": f"{data_name}_schema_record_{label}",
                "signal": torch.zeros(data_config["channels"], signal_length, dtype=torch.float32).numpy(),
                "label": label,
            }
        )

    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": output_mode},
        dataset_name=data_name,
        task_name=f"{data_name}_schema",
        in_memory=True,
    )


def load_tfm_model(
    data_name,
    device,
    tokenizer_checkpoint_path,
    *,
    use_classifier,
    classifier_checkpoint_path=None,
    emb_size=64,
    code_book_size=8192,
    trans_freq_encoder_depth=2,
    trans_temporal_encoder_depth=2,
    trans_decoder_depth=8,
    classifier_depth=4,
):
    sample_dataset = build_tfm_schema_dataset(data_name)
    model = TFMTokenizer(
        dataset=sample_dataset,
        emb_size=emb_size,
        code_book_size=code_book_size,
        trans_freq_encoder_depth=trans_freq_encoder_depth,
        trans_temporal_encoder_depth=trans_temporal_encoder_depth,
        trans_decoder_depth=trans_decoder_depth,
        use_classifier=use_classifier,
        classifier_depth=classifier_depth,
    )
    model = model.to(device)
    model.load_pretrained_weights(
        tokenizer_checkpoint_path=tokenizer_checkpoint_path,
        classifier_checkpoint_path=classifier_checkpoint_path,
        is_masked_training=False,
        strict=False,
        map_location=device,
    )
    model.eval()
    return model


def tfm_token_indices(model, signal):
    batch_size, num_channels, _ = signal.shape
    stft = get_stft_torch(signal)
    stft = rearrange(stft, "b c f t -> (b c) f t")
    signal_flat = rearrange(signal, "b c t -> (b c) t")
    _, token_indices, _ = model.tokenizer.tokenize(stft, signal_flat)
    return rearrange(token_indices, "(b c) t -> b c t", b=batch_size, c=num_channels)


class TFMClassifierBody(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        # Keep only the pre-head classifier modules here so the body and
        # classification_head remain disjoint parameter groups
        self.eeg_token_embedding = classifier.eeg_token_embedding
        self.channel_embed = classifier.channel_embed
        self.index = classifier.index
        self.temporal_pos_embed = classifier.temporal_pos_embed
        self.cls_token = classifier.cls_token
        self.LAT = classifier.LAT

    def forward(self, token_indices):
        x = self.eeg_token_embedding(token_indices)
        for channel_idx in range(x.shape[1]):
            used_channel_embed = (
                self.channel_embed(self.index[channel_idx])
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(x.size(0), -1, -1)
            )
            x[:, channel_idx] = self.temporal_pos_embed(x[:, channel_idx] + used_channel_embed)

        x = rearrange(x, "b c t e -> b (c t) e")
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.LAT(x)
        return x[:, 0]


class TFMCommonEncoder(nn.Module):
    """Wrap the pretrained TFM tokenizer and expose the quantized token sequence."""

    def __init__(
        self,
        data_name,
        device,
        tokenizer_checkpoint_path,
        emb_size=64,
        code_book_size=8192,
        trans_freq_encoder_depth=2,
        trans_temporal_encoder_depth=2,
        trans_decoder_depth=8,
    ):
        super().__init__()
        self.device = device
        self.output_dim = emb_size
        self.tokenizer = load_tfm_model(
            data_name=data_name,
            device=device,
            tokenizer_checkpoint_path=tokenizer_checkpoint_path,
            use_classifier=False,
            emb_size=emb_size,
            code_book_size=code_book_size,
            trans_freq_encoder_depth=trans_freq_encoder_depth,
            trans_temporal_encoder_depth=trans_temporal_encoder_depth,
            trans_decoder_depth=trans_decoder_depth,
        )

    def forward(self, signal, pos=None):
        # Match the shared encoder interface, since TFM does not use positions
        if signal.dim() == 2:
            signal = signal.unsqueeze(0)
        if signal.dim() != 3:
            raise ValueError(f"Unsupported TFM input shape: {tuple(signal.shape)}")

        batch_size, num_channels, _ = signal.shape
        stft = get_stft_torch(signal)
        stft = rearrange(stft, "b c f t -> (b c) f t")
        signal_flat = rearrange(signal, "b c t -> (b c) t")

        quant_out, _, _ = self.tokenizer.tokenizer.tokenize(stft, signal_flat)
        quant_out = rearrange(quant_out, "(b c) t e -> b (c t) e", b=batch_size, c=num_channels)
        # Return pooled [B, E] features so the classifier does not need
        # encoder-specific sequence pooling logic
        return quant_out.mean(dim=1)
