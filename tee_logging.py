import atexit
import sys
from pathlib import Path
from typing import TextIO


class TeeStream:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_tee_logging(log_path: str | Path, mirror_path: str | Path | None = None) -> Path:
    # Mirror stdout/stderr to both the terminal and a log file for this run
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mirror = Path(mirror_path) if mirror_path is not None else None
    if mirror is not None:
        mirror.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(path, "a", buffering=1)
    mirror_file = open(mirror, "a", buffering=1) if mirror is not None else None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    stdout_streams = [original_stdout, log_file]
    stderr_streams = [original_stderr, log_file]
    if mirror_file is not None:
        stdout_streams.append(mirror_file)
        stderr_streams.append(mirror_file)

    sys.stdout = TeeStream(*stdout_streams)
    sys.stderr = TeeStream(*stderr_streams)

    def restore_streams():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        if mirror_file is not None:
            mirror_file.close()

    atexit.register(restore_streams)
    return path
