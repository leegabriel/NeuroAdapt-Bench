import pickle

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset

from config import config
from utils.runtime import seed_everything


def preprocess_signal(
    signal,
    *,
    default_sampling_rate,
    resampling_rate,
    signal_len,
    normalization,
    drop_last_channel,
):
    signal = np.asarray(signal, dtype=np.float32)
    if drop_last_channel:
        signal = signal[:-1, :]
    # Always enforce the expected fixed window length so batching remains valid
    # even if a stored sample is slightly shorter or longer than intended
    target_length = int(resampling_rate * signal_len)
    if default_sampling_rate != resampling_rate or signal.shape[-1] != target_length:
        signal = resample(signal, target_length, axis=-1)
    if normalization == "95th_percentile":
        scale = np.quantile(
            np.abs(signal),
            q=0.95,
            axis=-1,
            method="linear",
            keepdims=True,
        )
        signal = signal / (scale + 1e-8)
    return signal


def split_patient_ids(patient_ids, train_ratio, val_ratio, seed):
    # Ported from pyhealth.datasets.splitter.split_by_patient so torch follows
    # the same patient-level train/val policy as the PyHealth backend
    patient_ids = list(sorted(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    num_patients = len(patient_ids)
    train_patient_ids = patient_ids[: int(num_patients * train_ratio)]
    val_patient_ids = patient_ids[
        int(num_patients * train_ratio) : int(num_patients * (train_ratio + val_ratio))
    ]
    return set(train_patient_ids), set(val_patient_ids)


def filter_files_by_patient(data_files, patient_ids, patient_id_getter):
    return [data_file for data_file in data_files if patient_id_getter(data_file) in patient_ids]


def build_patient_to_index(data_files, patient_id_getter):
    patient_to_index = {}
    for index, data_file in enumerate(data_files):
        patient_id = patient_id_getter(data_file)
        patient_to_index.setdefault(patient_id, []).append(index)
    return patient_to_index


class TUEVLoader(Dataset):
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    TRAIN_DIR = "processed_train"
    VAL_DIR = "processed_train"
    TEST_DIR = "processed_eval"

    def __init__(self, *, data_config, train_val_test, seed=None, **_):
        self.data_dir = data_config["processed_data_dir"]
        self.raw_data_dir = data_config["data_dir"]
        self.train_val_test = train_val_test
        self.default_sampling_rate = data_config["sampling_rate"]
        self.resampling_rate = data_config["resampling_rate"]
        self.signal_len = data_config["signal_len"]
        self.normalization = data_config["normalization"]
        self.drop_last_channel = data_config["drop_last_channel"]
        self.signal_key = data_config["signal_key"]
        self.label_key = data_config["label_key"]
        if train_val_test == "train":
            split_dir = self.TRAIN_DIR
        elif train_val_test == "val":
            split_dir = self.VAL_DIR
        elif train_val_test == "test":
            split_dir = self.TEST_DIR
        else:
            raise ValueError(f"Unsupported train_val_test: {train_val_test}")
        self.data_files = sorted((self.data_dir / split_dir).iterdir())
        if train_val_test in {"train", "val"}:
            train_patient_ids, val_patient_ids = split_patient_ids(
                self.train_pool_patient_ids(),
                train_ratio=self.TRAIN_RATIO,
                val_ratio=self.VAL_RATIO,
                seed=seed,
            )
            patient_ids = train_patient_ids if train_val_test == "train" else val_patient_ids
            self.data_files = filter_files_by_patient(
                self.data_files,
                patient_ids,
                self.train_patient_id_from_file,
            )
        else:
            self.data_files = filter_files_by_patient(
                self.data_files,
                self.test_pool_patient_ids(),
                self.test_patient_id_from_file,
            )
        patient_id_getter = self.test_patient_id_from_file if train_val_test == "test" else self.train_patient_id_from_file
        self.patient_to_index = build_patient_to_index(self.data_files, patient_id_getter)

    def train_pool_patient_ids(self):
        return {path.name for path in (self.raw_data_dir / "train").iterdir() if path.is_dir()}

    def test_pool_patient_ids(self):
        return {path.name for path in (self.raw_data_dir / "eval").iterdir() if path.is_dir()}

    def train_patient_id_from_file(self, data_file):
        return data_file.name.split("_")[0]

    def test_patient_id_from_file(self, data_file):
        return data_file.name.split("_")[1]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        with self.data_files[index].open("rb") as handle:
            signal_data = pickle.load(handle)
        signal = preprocess_signal(
            signal_data[self.signal_key],
            default_sampling_rate=self.default_sampling_rate,
            resampling_rate=self.resampling_rate,
            signal_len=self.signal_len,
            normalization=self.normalization,
            drop_last_channel=self.drop_last_channel,
        )
        label = int(np.asarray(signal_data[self.label_key]).reshape(-1)[0]) - 1
        return {
            "signal": torch.as_tensor(signal, dtype=torch.float32),
            "label": torch.as_tensor(label),
        }


class TUABLoader(Dataset):
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    TRAIN_DIR = "train"
    VAL_DIR = "train"
    TEST_DIR = "test"

    def __init__(self, *, data_config, train_val_test, seed=None, **_):
        self.data_dir = data_config["processed_data_dir"]
        self.raw_data_dir = data_config["data_dir"]
        self.train_val_test = train_val_test
        self.default_sampling_rate = data_config["sampling_rate"]
        self.resampling_rate = data_config["resampling_rate"]
        self.signal_len = data_config["signal_len"]
        self.normalization = data_config["normalization"]
        self.drop_last_channel = data_config["drop_last_channel"]
        self.signal_key = data_config["signal_key"]
        self.label_key = data_config["label_key"]
        if train_val_test == "train":
            split_dir = self.TRAIN_DIR
        elif train_val_test == "val":
            split_dir = self.VAL_DIR
        elif train_val_test == "test":
            split_dir = self.TEST_DIR
        else:
            raise ValueError(f"Unsupported train_val_test: {train_val_test}")
        self.data_files = sorted((self.data_dir / split_dir).iterdir())
        if train_val_test in {"train", "val"}:
            train_patient_ids, val_patient_ids = split_patient_ids(
                self.train_pool_patient_ids(),
                train_ratio=self.TRAIN_RATIO,
                val_ratio=self.VAL_RATIO,
                seed=seed,
            )
            patient_ids = train_patient_ids if train_val_test == "train" else val_patient_ids
            self.data_files = filter_files_by_patient(
                self.data_files,
                patient_ids,
                self.patient_id_from_file,
            )
        else:
            self.data_files = filter_files_by_patient(
                self.data_files,
                self.test_pool_patient_ids(),
                self.patient_id_from_file,
            )
        self.patient_to_index = build_patient_to_index(self.data_files, self.patient_id_from_file)

    def train_pool_patient_ids(self):
        patient_ids = set()
        for label_name in ("normal", "abnormal"):
            for edf_file in (self.raw_data_dir / "train" / label_name / "01_tcp_ar").glob("*.edf"):
                patient_ids.add(edf_file.stem.split("_")[0])
        return patient_ids

    def test_pool_patient_ids(self):
        patient_ids = set()
        for label_name in ("normal", "abnormal"):
            for edf_file in (self.raw_data_dir / "eval" / label_name / "01_tcp_ar").glob("*.edf"):
                patient_ids.add(edf_file.stem.split("_")[0])
        return patient_ids

    def patient_id_from_file(self, data_file):
        return data_file.name.split("_")[0]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        with self.data_files[index].open("rb") as handle:
            signal_data = pickle.load(handle)
        signal = preprocess_signal(
            signal_data[self.signal_key],
            default_sampling_rate=self.default_sampling_rate,
            resampling_rate=self.resampling_rate,
            signal_len=self.signal_len,
            normalization=self.normalization,
            drop_last_channel=self.drop_last_channel,
        )
        label = int(np.asarray(signal_data[self.label_key]).reshape(-1)[0])
        return {
            "signal": torch.as_tensor(signal, dtype=torch.float32),
            "label": torch.as_tensor(label),
        }


class CHBMITLoader(Dataset):
    TRAIN_DIR = "train"
    VAL_DIR = "val"
    TEST_DIR = "test"

    def __init__(self, *, data_config, train_val_test, **_):
        self.data_dir = data_config["data_dir"]
        self.train_val_test = train_val_test
        self.default_sampling_rate = data_config["sampling_rate"]
        self.resampling_rate = data_config["resampling_rate"]
        self.signal_len = data_config["signal_len"]
        self.normalization = data_config["normalization"]
        self.drop_last_channel = data_config["drop_last_channel"]
        self.signal_key = data_config["signal_key"]
        self.label_key = data_config["label_key"]
        if train_val_test == "train":
            split_dir = self.TRAIN_DIR
        elif train_val_test == "val":
            split_dir = self.VAL_DIR
        elif train_val_test == "test":
            split_dir = self.TEST_DIR
        else:
            raise ValueError(f"Unsupported train_val_test: {train_val_test}")
        self.data_files = sorted((self.data_dir / split_dir).iterdir())
        self.patient_to_index = build_patient_to_index(self.data_files, self.patient_id_from_file)

    def patient_id_from_file(self, data_file):
        return data_file.stem.split("_")[0]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        with self.data_files[index].open("rb") as handle:
            signal_data = pickle.load(handle)
        signal = preprocess_signal(
            signal_data[self.signal_key],
            default_sampling_rate=self.default_sampling_rate,
            resampling_rate=self.resampling_rate,
            signal_len=self.signal_len,
            normalization=self.normalization,
            drop_last_channel=self.drop_last_channel,
        )
        label = int(np.asarray(signal_data[self.label_key]).reshape(-1)[0])
        return {
            "signal": torch.as_tensor(signal, dtype=torch.float32),
            "label": torch.as_tensor(label),
        }


class EarEEGLoader(Dataset):
    TRAIN_DIR = "train"
    VAL_DIR = "val"
    TEST_DIR = "test"

    def __init__(self, *, data_config, train_val_test, **_):
        self.data_dir = data_config["data_dir"]
        self.train_val_test = train_val_test
        self.default_sampling_rate = data_config["sampling_rate"]
        self.resampling_rate = data_config["resampling_rate"]
        self.signal_len = data_config["signal_len"]
        self.normalization = data_config["normalization"]
        self.drop_last_channel = data_config["drop_last_channel"]
        self.signal_key = data_config["signal_key"]
        self.label_key = data_config["label_key"]
        if train_val_test == "train":
            split_dir = self.TRAIN_DIR
        elif train_val_test == "val":
            split_dir = self.VAL_DIR
        elif train_val_test == "test":
            split_dir = self.TEST_DIR
        else:
            raise ValueError(f"Unsupported train_val_test: {train_val_test}")
        self.data_files = sorted((self.data_dir / split_dir).iterdir())
        self.patient_to_index = build_patient_to_index(self.data_files, self.patient_id_from_file)

    def patient_id_from_file(self, data_file):
        return data_file.stem.split("_")[0]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        with self.data_files[index].open("rb") as handle:
            signal_data = pickle.load(handle)
        signal = preprocess_signal(
            signal_data[self.signal_key],
            default_sampling_rate=self.default_sampling_rate,
            resampling_rate=self.resampling_rate,
            signal_len=self.signal_len,
            normalization=self.normalization,
            drop_last_channel=self.drop_last_channel,
        )
        label = int(np.asarray(signal_data[self.label_key]).reshape(-1)[0])
        return {
            "signal": torch.as_tensor(signal, dtype=torch.float32),
            "label": torch.as_tensor(label),
        }


class SleepEDF78Loader(Dataset):
    TRAIN_DIR = "train"
    VAL_DIR = "val"
    TEST_DIR = "test"

    def __init__(self, *, data_config, train_val_test, **_):
        self.data_dir = data_config["data_dir"]
        self.train_val_test = train_val_test
        self.default_sampling_rate = data_config["sampling_rate"]
        self.resampling_rate = data_config["resampling_rate"]
        self.signal_len = data_config["signal_len"]
        self.normalization = data_config["normalization"]
        self.drop_last_channel = data_config["drop_last_channel"]
        if train_val_test == "train":
            split_dir = self.TRAIN_DIR
        elif train_val_test == "val":
            split_dir = self.VAL_DIR
        elif train_val_test == "test":
            split_dir = self.TEST_DIR
        else:
            raise ValueError(f"Unsupported train_val_test: {train_val_test}")
        self.data_files = sorted((self.data_dir / split_dir).iterdir())
        self.patient_to_index = build_patient_to_index(self.data_files, self.patient_id_from_file)

    def patient_id_from_file(self, data_file):
        parts = data_file.stem.split("_")
        return "_".join(parts[:2])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        signal_data = np.load(self.data_files[index], allow_pickle=True)
        signal = preprocess_signal(
            signal_data[0],
            default_sampling_rate=self.default_sampling_rate,
            resampling_rate=self.resampling_rate,
            signal_len=self.signal_len,
            normalization=self.normalization,
            drop_last_channel=self.drop_last_channel,
        )
        label = int(signal_data[1][0])
        return {
            "signal": torch.as_tensor(signal, dtype=torch.float32),
            "label": torch.as_tensor(label),
        }

def create_dataloader(*, data_name, train_val_test, batch_size, **kwargs):
    data_config = kwargs.get("data_config", getattr(config.datasets, data_name.upper()))
    seed_everything(kwargs["seed"])

    if data_name == "tuev":
        dataset = TUEVLoader(
            data_config=data_config,
            train_val_test=train_val_test,
            seed=kwargs["seed"],
        )
    elif data_name == "tuab":
        dataset = TUABLoader(
            data_config=data_config,
            train_val_test=train_val_test,
            seed=kwargs["seed"],
        )
    elif data_name == "chbmit":
        dataset = CHBMITLoader(
            data_config=data_config,
            train_val_test=train_val_test,
        )
    elif data_name == "eareeg":
        dataset = EarEEGLoader(
            data_config=data_config,
            train_val_test=train_val_test,
        )
    elif data_name == "sleep_edf_78":
        dataset = SleepEDF78Loader(
            data_config=data_config,
            train_val_test=train_val_test,
        )
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train_val_test == "train"),
        num_workers=8,
    )
