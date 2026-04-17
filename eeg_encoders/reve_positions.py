from functools import lru_cache

from config import config
from eeg_encoders.positions import resolve_positions


@lru_cache(maxsize=None)
def data_positions(data_name):
    reve_electrodes = config.models.reve.ELECTRODES[data_name]
    reve_position_aliases = config.models.reve.ELECTRODE_ALIASES.get(data_name, {})
    return resolve_positions(
        reve_electrodes,
        config.models.reve.POSITIONS_DIR,
        aliases=reve_position_aliases,
    )
