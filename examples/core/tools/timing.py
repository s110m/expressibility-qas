import datetime
import os
import time

from torchpack.utils import io
from torchpack.utils.logging import logger

__all__ = ['StepTimer']


class StepTimer:
    """Context manager that times a QuantumNAS pipeline step (supercircuit
    training, evolutionary/NSGA-II search, Pareto-front scoring, subcircuit
    training, evaluation) and persists the elapsed wall-clock time to
    `<out_dir>/<filename>`.

    Only train.py's run dir is wired into torchpack's file-logging sink
    (via `set_run_dir`) -- the other driver scripts are not, so without an
    explicit record here their timings would only ever exist in console
    scrollback. Records are written on both success and exception (labeled
    via `status`), since long training/search jobs can crash partway
    through and the partial duration is still useful.
    """

    def __init__(self, step_name, out_dir, filename='timing.json', extra=None):
        self.step_name = step_name
        self.out_dir = out_dir
        self.filename = filename
        self.extra = extra or {}

    def __enter__(self):
        self.start_time = time.time()
        self._start_perf = time.perf_counter()
        logger.info(f'[timing] "{self.step_name}" started.')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_seconds = time.perf_counter() - self._start_perf
        end_time = time.time()
        # SystemExit(0) (e.g. an early `exit(0)` short-circuit) is a normal
        # completion, not a failure.
        is_clean_exit = exc_type is None or (
            exc_type is SystemExit and exc_val.code in (None, 0))
        status = 'completed' if is_clean_exit else 'failed'

        logger.info(
            f'[timing] "{self.step_name}" {status} in '
            f'{datetime.timedelta(seconds=round(elapsed_seconds))} '
            f'({elapsed_seconds:.2f}s).')

        record = {
            'step': self.step_name,
            'status': status,
            'start_time': datetime.datetime.fromtimestamp(
                self.start_time).isoformat(),
            'end_time': datetime.datetime.fromtimestamp(end_time).isoformat(),
            'elapsed_seconds': elapsed_seconds,
            **self.extra,
        }
        try:
            io.save(os.path.join(self.out_dir, self.filename), record)
        except Exception as e:
            logger.warning(f'[timing] Could not save timing record: {e}')

        return False
