"""
:class:`~allennlp.common.tqdm.Tqdm` wraps tqdm so we can add configurable
global defaults for certain tqdm parameters.
"""

from tqdm.auto import tqdm as _tqdm

# This is neccesary to stop tqdm from hanging
# when exceptions are raised inside iterators.
# It should have been fixed in 4.2.1, but it still
# occurs.
# TODO(Mark): Remove this once tqdm cleans up after itself properly.
# https://github.com/tqdm/tqdm/issues/469
_tqdm.monitor_interval = 0

"""iterable = None, desc = None, total = None, leave = True,
file = None, ncols = None, mininterval = 0.1, maxinterval = 10.0,
miniters = None, ascii = None, disable = False, unit = 'it',
unit_scale = False, dynamic_ncols = False, smoothing = 0.3,
bar_format = None, initial = 0, position = None, postfix = None,
unit_divisor = 1000, write_bytes = None, lock_args = None,
gui = False,
"""


class Tqdm(_tqdm):
    # These defaults are the same as the argument defaults in tqdm.
    default_mininterval: float = 0.1
    default_dynamic_ncols: bool = True
    default_bar_format: str = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

    # l_bar='{desc}: {percentage:3.0f}%|' and
    # r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
    #   '{rate_fmt}{postfix}]'
    # Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
    #   percentage, elapsed, elapsed_s, ncols, desc, unit,
    #   rate, rate_fmt, rate_noinv, rate_noinv_fmt,
    #   rate_inv, rate_inv_fmt, postfix, unit_divisor,
    #   remaining, remaining_s.

    def __init__(self, iterable, desc=None, **kwargs):
        bar_format: str = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        super().__init__(iterable, desc, leave=False, dynamic_ncols=True,
                         bar_format=bar_format, **kwargs)

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {
            'mininterval': Tqdm.default_mininterval,
            'dynamic_ncols': Tqdm.default_dynamic_ncols,
            **kwargs
        }
        return _tqdm(*args, **new_kwargs)

    # @staticmethod
    # def write(str, file=None, end="\n", nolock=False):
    #     _tqdm.write(str, file, end, nolock)
