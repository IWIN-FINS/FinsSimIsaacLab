#!/usr/bin/env python3
"""Replay a registered FinsSim Isaac task through Isaac Lab's SKRL runner."""

from skrl_runner import run


if __name__ == "__main__":
    run("play")
