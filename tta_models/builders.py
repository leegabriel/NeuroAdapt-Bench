from pathlib import Path

import torch

from config import config
from eeg_encoders.tfm import TFMCommonEncoder
from tta_models.common_classifier import CommonEEGClassifier


def _data_cfg(data_name):
    return getattr(config.datasets, data_name.upper())


def _load_checkpoint_if_present(model, checkpoint_path, map_location):
    if checkpoint_path is None:
        return model

    checkpoint = Path(checkpoint_path)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=map_location))
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return model


def build_cbramod_model(
    device,
    num_classes,
    data_name,
    checkpoint_path=None,
    projection_dim=128,
    dropout=0.1,
):
    from eeg_encoders.cbramod import CBraMod_Wrapper

    encoder = CBraMod_Wrapper(
        device=device,
        # The shared classifier expects pooled [B, E] features from CBraMod
        cls_only=True,
        target_length=int(_data_cfg(data_name)["signal_len"] * 200),
        pretrained_weights_path=config.models.cbramod.PRETRAINED_CKPT,
        input_scale=config.models.cbramod.INPUT_SCALE,
    )
    if encoder.cls_only is not True:
        raise ValueError("CBraMod benchmark path requires cls_only=True to return pooled [B, E] features.")
    model = CommonEEGClassifier(
        encoder=encoder,
        feature_dim=encoder.output_dim,
        num_classes=num_classes,
        projection_dim=projection_dim,
        dropout=dropout,
    )
    model = model.to(device)
    return _load_checkpoint_if_present(model, checkpoint_path, map_location=device)


def build_tfm_common_model(
    data_name,
    device,
    num_classes,
    tokenizer_checkpoint_path,
    checkpoint_path=None,
    projection_dim=128,
    dropout=0.1,
    emb_size=64,
    code_book_size=8192,
    trans_freq_encoder_depth=2,
    trans_temporal_encoder_depth=2,
    trans_decoder_depth=8,
):
    encoder = TFMCommonEncoder(
        data_name=data_name,
        device=device,
        tokenizer_checkpoint_path=tokenizer_checkpoint_path,
        emb_size=emb_size,
        code_book_size=code_book_size,
        trans_freq_encoder_depth=trans_freq_encoder_depth,
        trans_temporal_encoder_depth=trans_temporal_encoder_depth,
        trans_decoder_depth=trans_decoder_depth,
    )
    model = CommonEEGClassifier(
        encoder=encoder,
        feature_dim=encoder.output_dim,
        num_classes=num_classes,
        projection_dim=projection_dim,
        dropout=dropout,
    )
    model = model.to(device)
    return _load_checkpoint_if_present(model, checkpoint_path, map_location=device)


def build_reve_model(
    model_dir,
    device,
    num_classes,
    checkpoint_path=None,
    projection_dim=128,
    dropout=0.1,
):
    from eeg_encoders.reve import REVEEncoder

    encoder = REVEEncoder(model_dir=model_dir, device=device, pooled=True)
    model = CommonEEGClassifier(
        encoder=encoder,
        feature_dim=encoder.output_dim,
        num_classes=num_classes,
        projection_dim=projection_dim,
        dropout=dropout,
    )
    model = model.to(device)
    return _load_checkpoint_if_present(model, checkpoint_path, map_location=device)


def build_reve_base_model(device, num_classes, checkpoint_path=None, projection_dim=128, dropout=0.1, **_):
    return build_reve_model(
        model_dir=config.models.reve.BASE_DIR,
        device=device,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        projection_dim=projection_dim,
        dropout=dropout,
    )


def build_reve_large_model(device, num_classes, checkpoint_path=None, projection_dim=128, dropout=0.1, **_):
    return build_reve_model(
        model_dir=config.models.reve.LARGE_DIR,
        device=device,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        projection_dim=projection_dim,
        dropout=dropout,
    )


def model_num_classes(experiment, encoder, data_name):
    return getattr(config.datasets, data_name.upper())["classes"]


MODEL_BUILDERS = {
    ("common", "cbramod"): build_cbramod_model,
    ("common", "tfm"): build_tfm_common_model,
    ("common", "reve_base"): build_reve_base_model,
    ("common", "reve_large"): build_reve_large_model,
}


def build_model(experiment, encoder, data_name, device, checkpoint_path=None, projection_dim=128, dropout=0.1):
    num_classes = model_num_classes(experiment, encoder, data_name)
    kwargs = {
        "device": device,
        "data_name": data_name,
        "num_classes": num_classes,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    if experiment == "common":
        kwargs["projection_dim"] = projection_dim
        kwargs["dropout"] = dropout
    if encoder in {"tfm"}:
        kwargs["tokenizer_checkpoint_path"] = config.models.tfm.TOKENIZER_CKPT
    model = MODEL_BUILDERS[(experiment, encoder)](**kwargs)
    model.data_name = data_name
    return model
