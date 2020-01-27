from __future__ import print_function

import argparse


def train(**args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    arguments = args.__dict__
    train(**arguments)
