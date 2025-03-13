import os
import sys
import time
from functools import reduce
import operator
import torch
from logging import getLogger

logger = getLogger(__name__)


_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_SIGNAL_HANDLER = None
_GLOBAL_MEMORY_BUFFER = None
_GLOBAL_SLOT_HANDLE = None


def get_global_memory_buffer():
    _ensure_var_is_initialized(_GLOBAL_MEMORY_BUFFER, "global memory buffer")
    return _GLOBAL_MEMORY_BUFFER


def get_slot_handle():
    # _ensure_var_is_initialized(_GLOBAL_SLOT_HANDLE, "slot_handle")
    return _GLOBAL_SLOT_HANDLE


def set_global_variables(global_args=None):
    set_global_args(global_args)
    _set_timers()
    if global_args is not None:
        _set_slot_handle(
            global_args.infer.max_reqs,
            global_args.infer.pp_size,
            global_args.infer.cache_type,
        )


def _set_slot_handle(max_reqs, pp_size, cache_type):
    global _GLOBAL_SLOT_HANDLE
    _ensure_var_is_not_initialized(_GLOBAL_SLOT_HANDLE, "slot_handle")
    if cache_type == "skew" and pp_size > 1:
        _GLOBAL_SLOT_HANDLE = SlotHandle(max_reqs, pp_size)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, "tensorboard writer")

    if (
        hasattr(args, "tensorboard_dir")
        and args.tensorboard_dir
        and args.rank == (args.world_size - 1)
    ):
        try:
            from torch.utils.tensorboard import SummaryWriter

            logger.info("> setting tensorboard ...")
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir, max_queue=args.tensorboard_queue_size
            )
        except ModuleNotFoundError:
            logger.warning(
                "TensorBoard writing requested but is not "
                "available (are you using PyTorch 1.1.0 or later?), "
                "no TensorBoard logs will be written.",
                flush=True,
            )


def get_global_args():
    _ensure_var_is_initialized(_GLOBAL_ARGS, "global args")
    return _GLOBAL_ARGS


def set_global_args(args):
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, "global args")
    _GLOBAL_ARGS = args


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS


def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
    _GLOBAL_TIMERS = Timers()


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    _ensure_var_is_not_initialized(_GLOBAL_MEMORY_BUFFER, "global memory buffer")
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()
        self.cnt = 0

    def start(self):
        # return
        """Start the timer."""
        assert not self.started_, "timer has already been started"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        # return
        """Stop the timer."""
        assert self.started_, "timer is not started"
        torch.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False
        self.cnt += 1

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False
        self.cnt = 0

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + "-time", value, iteration)

    def log(self, names=[], normalizer=1.0, reset=True):
        """Log a group of timers."""
        if len(names) == 0:
            names = self.timers.keys()
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            cnt = self.timers[name].cnt
            if cnt == 0:
                continue
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f} {} {:.2f}".format(
                name, elapsed_time, cnt, elapsed_time / cnt
            )
        logger.info(string)


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


class SlotHandle:
    """
    split max_reqs to micro_batch size
    max_req = 10, pp_size = 3, self.slots_size = [4, 3, 3]
    """

    def __init__(self, max_reqs, pp_size):
        self.slots_size = self.split_slots(max_reqs, pp_size)
        self.num_slots = len(self.slots_size)
        self.slot_idx = 0
        self.slot_start_idx = []
        self.slot_end_idx = []

        res = [0]
        for value in self.slots_size:
            res.append(res[-1] + value)
        self.slot_start_idx = res[:-1]
        self.slot_end_idx = res[1:]

    def split_slots(self, total, parts):
        result = [0] * min(total, parts)
        for i in range(total):
            result[i % parts] += 1
        return result

    def get_slot_size(self, idx):
        return self.slots_size[idx]

    def set_slot_idx(self, idx):
        self.slot_idx = idx

    def get_slot_idx(self):
        return self.slot_idx

    def get_slot_start_end_idx(self, idx):
        return self.slot_start_idx[idx], self.slot_end_idx[idx]

    def get_current_slot_start_end_idx(self):
        return self.get_slot_start_end_idx(self.slot_idx)
