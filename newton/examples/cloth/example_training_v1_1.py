# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deprecated compatibility shim for :mod:`newton.examples.cloth.example_training_v1_2`.

Importing this module emits ``DeprecationWarning``. Prefer ``example_training_v1_2`` and CLI
``python -m newton.examples training_v1_2``.
"""

from __future__ import annotations

import warnings

from newton.examples.cloth.example_training_v1_2 import *  # noqa: F403

from newton.examples.cloth import example_training_v1_2 as _v12

warnings.warn(
    (
        "newton.examples.cloth.example_training_v1_1 is deprecated; use "
        "example_training_v1_2 (CLI: python -m newton.examples training_v1_2)."
    ),
    DeprecationWarning,
    stacklevel=2,
)

TrainingDemoV1_1Api = _v12.TrainingDemoV1_2Api

run_training_v1_1_demo = _v12.run_training_v1_2_demo

__all__ = list(_v12.__all__)
__all__.append("TrainingDemoV1_1Api")
__all__.append("run_training_v1_1_demo")


if __name__ == "__main__":
    run_training_v1_1_demo(parser_defaults={"num_frames": 3850})
