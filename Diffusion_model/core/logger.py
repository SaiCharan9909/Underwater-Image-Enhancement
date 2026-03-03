import os
import logging
from datetime import datetime
import yaml


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def parse(args):
    """
    Simplified config parser for inference-based usage.
    """
    phase = args.phase
    opt_path = args.config

    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)

    opt['phase'] = phase

    # Optional GPU handling (single device only)
    if hasattr(args, "gpu_ids") and args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
        opt['gpu_ids'] = [int(i) for i in gpu_ids.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print("Using GPU:", gpu_ids)

    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    """Convert dict to readable string for logging"""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=True):
    """
    Basic logger setup for inference.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    log_file = os.path.join(root, f"{phase}.log")
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger