from pathlib import Path


# Paths and output locations
class PathConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent
    OUTPUT_DIR = PROJECT_ROOT / "output"
    RUNS_DIR = PROJECT_ROOT / "runs"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    EXPERIMENTS_SUMMARY_DIR = REPORTS_DIR / "experiments_summary"
    EXPERIMENTS_SUMMARY_CSV = EXPERIMENTS_SUMMARY_DIR / "results.csv"


# Training defaults used by both fine-tuning and benchmarking
class TrainingConfig:
    SEEDS = (5, 6, 3, 5635, 10996)
    DEFAULT_SEED = SEEDS[0]
    BATCH_SIZE = 32
    BACKEND = "pytorch"


# Registered experiment namespaces
class ExperimentConfig:
    NAMES = ("common",)
    EXP_1_DIR = PathConfig.OUTPUT_DIR / "exp_1"
    DIRS = {
        "common": EXP_1_DIR,
    }


# Foundation model checkpoints and model-specific constants
class TFMModelConfig:
    TOKENIZER_CKPT = Path(
        "/projects/illinois/eng/cs/jimeng/gjlee4/code/TFM-Tokenizer/pretrained_weigths/"
        "multiple_dataset_settings/Pretrained_tfm_tokenizer_2x2x8/"
        "tfm_tokenizer_last.pth"
    )


class CBraModModelConfig:
    PRETRAINED_CKPT = Path("/projects/illinois/eng/cs/jimeng/gjlee4/code/CBraMod/pretrained_weights.pth")
    # CBraMod expects raw EEG amplitudes scaled down before they are passed into
    # the pretrained backbone
    INPUT_SCALE = 100.0


class ReveModelConfig:
    # Local checkpoints for the two REVE model sizes
    BASE_DIR = Path("/projects/illinois/eng/cs/jimeng/gjlee4/code/reve-base")
    LARGE_DIR = Path("/projects/illinois/eng/cs/jimeng/gjlee4/code/reve-large")
    # Shared bank of electrode coordinates used to build REVE position tensors
    POSITIONS_DIR = Path("/projects/illinois/eng/cs/jimeng/gjlee4/code/reve-positions")
    # Channel names that REVE position lookup should use for each dataset
    ELECTRODES = {
        "tuev": [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        ],
        "tuab": [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        ],
        "chbmit": [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        ],
        "eareeg": ["RB", "RT", "LB", "LT"],
        "sleep_edf_78": ["Fpz-Cz", "Pz-Oz"],
    }
    # Dataset-specific name rewrites needed before querying the REVE position bank
    ELECTRODE_ALIASES = {
        "eareeg": {"RB": "A2", "RT": "T8", "LB": "A1", "LT": "T7"},
    }


# Benchmark-supported foundation models
class ModelConfig:
    NAMES = ("cbramod", "tfm", "reve_base", "reve_large")
    cbramod = CBraModModelConfig()
    tfm = TFMModelConfig()
    reve = ReveModelConfig()


# Dataset registry for the benchmark suite
class DatasetConfig:
    NAMES = ("tuev", "tuab", "chbmit", "eareeg", "sleep_edf_78")
    TUEV = {
            "data_dir": Path("/projects/illinois/eng/cs/jimeng/gjlee4/data/srv/gjlee4/TUH/tuh_eeg_events/v2.0.0/edf"),
            "processed_data_dir": Path("/projects/illinois/eng/cs/jimeng/gjlee4/data/srv/gjlee4/TUH/tuh_eeg_events/v2.0.0/edf"),
            "task": "multiclass",
            "classes": 6,
            "sampling_rate": 200,
            "resampling_rate": 200,
            "signal_len": 5.0,
            "channels": 16,
            "normalization": "95th_percentile",
            "signal_key": "signal",
            "label_key": "label",
            "drop_last_channel": False,
    }
    TUAB = {
            "data_dir": Path("/projects/illinois/eng/cs/jimeng/gjlee4/data/srv/gjlee4/TUH/tuh_eeg_abnormal/v3.0.0/edf"),
            "processed_data_dir": Path("/projects/illinois/eng/cs/jimeng/gjlee4/data/srv/gjlee4/TUH/tuh_eeg_abnormal/v3.0.0/edf/processed"),
            "task": "binary",
            "classes": 2,
            "sampling_rate": 200,
            "resampling_rate": 200,
            "signal_len": 10.0,
            "channels": 16,
            "normalization": "95th_percentile",
            "signal_key": "X",
            "label_key": "y",
            "drop_last_channel": False,
    }
    CHBMIT = {
            "data_dir": Path("/projects/illinois/eng/cs/jimeng/gjlee4/data/srv/gjlee4/chbmit/1.0.0/clean_segments"),
            "fmt": "pickle_xy",
            "task": "binary",
            "classes": 2,
            "sampling_rate": 256,
            "resampling_rate": 256,
            "signal_len": 10.0,
            "channels": 16,
            "normalization": "95th_percentile",
            "signal_key": "X",
            "label_key": "y",
            "drop_last_channel": False,
    }
    EAREEG = {
            "data_dir": Path("/projects/illinois/eng/cs/jimeng/gjlee4/data/srv/gjlee4/earEEG/processed_data"),
            "fmt": "pickle_x_label",
            "task": "multiclass",
            "classes": 6,
            "sampling_rate": 250,
            "resampling_rate": 250,
            "signal_len": 30.0,
            "channels": 4,
            "normalization": "95th_percentile",
            "signal_key": "X",
            "label_key": "label",
            "drop_last_channel": True,
    }
    SLEEP_EDF_78 = {
            "data_dir": Path("/projects/illinois/eng/cs/jimeng/gjlee4/data/srv/gjlee4/sleep_edf_78"),
            "fmt": "npy_pair",
            "task": "multiclass",
            "classes": 5,
            "sampling_rate": 100,
            "resampling_rate": 100,
            "signal_len": 30.0,
            "channels": 2,
            "normalization": "95th_percentile",
            "signal_key": "signal",
            "label_key": "label",
            "drop_last_channel": False,
    }


class Config:
    paths = PathConfig()
    training = TrainingConfig()
    experiments = ExperimentConfig()
    models = ModelConfig()
    datasets = DatasetConfig()


config = Config()
