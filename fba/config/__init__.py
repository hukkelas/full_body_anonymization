import argparse
from .base import Config

def default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser


def default_infer_parser() -> argparse.ArgumentParser():
    parser = default_parser()
    parser.add_argument("-t", "--truncation_value", default=0, type=float)
    parser.add_argument("-g", "--global_step", default=None, type=int)
    return parser
