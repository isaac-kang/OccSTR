import errno
import os
import sys
import logging
import functools

logger_initialized = {}


class StaleResistantFileHandler(logging.FileHandler):
    """FileHandler that reopens the stream on NFS stale handles / bad FDs.

    Why: training runs write to log files on NFS over many hours. If the NFS
    server hiccups, the open file handle goes stale (ESTALE, errno 116) and
    every subsequent emit/flush raises forever, dropping all later log lines.
    """

    _RECOVERABLE = {errno.ESTALE, errno.EBADF, errno.EIO, errno.ENOENT}

    def _reopen(self):
        try:
            if self.stream is not None:
                self.stream.close()
        except Exception:
            pass
        self.stream = None
        self.stream = self._open()

    def _write(self, record):
        if self.stream is None:
            self.stream = self._open()
        msg = self.format(record)
        self.stream.write(msg + self.terminator)
        self.stream.flush()

    def emit(self, record):
        try:
            self._write(record)
        except RecursionError:
            raise
        except OSError as e:
            if e.errno not in self._RECOVERABLE:
                self.handleError(record)
                return
            try:
                self._reopen()
                self._write(record)
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


@functools.lru_cache()
def get_logger(name="openrec", log_file=None, log_level=logging.DEBUG):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    if log_file is not None and rank == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = StaleResistantFileHandler(log_file, "a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    logger.propagate = False
    return logger
