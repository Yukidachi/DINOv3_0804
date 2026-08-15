"""A1-FT server entry point.

A1-FT is the end-to-end supervised fine-tuning counterpart of A1.  It starts
from the best A1 frozen-backbone probe checkpoint, unfreezes the complete
MobileNetV2 backbone, and runs the same 80,000-step pixel-CE protocol as
A0-FT/S2-0:

    source      : A1 probe (A1 fixed 1x1-Conv feature pretraining)
    trainable   : MobileNetV2 backbone + R-ASPP head
    loss        : hard-label pixel cross-entropy only; no KD loss
    optimizer   : SGD(lr=0.01, momentum=0.9, weight_decay=1e-4) + poly(0.9)

The A1 and A0 student checkpoints have the same MobileNetV2+R-ASPP state
layout.  The shared A0-FT implementation therefore provides the training,
DDP, resume, evaluation, checkpoint, and ordered-teardown logic; this entry
point only changes the experiment identity and default A1 source paths.  The
resulting artifacts are still written under the shared A-group output root,
but in an independent ``A1-FT`` run directory.

Example:

    torchrun --standalone --nproc_per_node=2 dino_a1_ft_server.py \\
        --seed 42 --batch-size 2 --global-batch-size 8 \\
        --num-workers 8 --multiprocessing-context spawn \\
        --no-pin-memory --persistent-workers

For a Windows single-process smoke test:

    python -B dino_a1_ft_server.py --device cuda --smoke-test \\
        --batch-size 1 --global-batch-size 1 --num-workers 0 \\
        --no-persistent-workers --no-pin-memory --no-amp
"""

from __future__ import annotations

from pathlib import Path

import dino_a0_ft_server as _a0_ft


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "A_MobileNetV2_RASPP_server"


# dino_a0_ft_server deliberately keeps the A1/A5 adaptation points as module
# globals.  Patch those globals before invoking its parser/training entry
# point so every dynamically-read field (paths, payload labels, resume checks,
# final reports, and script hash) consistently identifies A1-FT.
_a0_ft.__dict__.update(
    {
        "__file__": str(Path(__file__).resolve()),
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "EXPERIMENT": "A1-FT",
        "RUN_SUBDIR": "A1-FT",
        "DEFAULT_SOURCE_EXPERIMENT": "A1",
        "ARTIFACT_TYPE_FINETUNE": "a1_ft_mobilenetv2_raspp",
    }
)

# Keep the same useful module-level API as the concrete A0-FT entry point for
# checkpoint inspection and small diagnostic scripts.  These function objects
# resolve their globals in ``dino_a0_ft_server``; the identity patch above is
# therefore applied to them as well.
EXPERIMENT = "A1-FT"
RUN_SUBDIR = "A1-FT"
DEFAULT_SOURCE_EXPERIMENT = "A1"
ARTIFACT_TYPE_FINETUNE = "a1_ft_mobilenetv2_raspp"
MODEL_NAME = _a0_ft.MODEL_NAME
NUM_CLASSES = _a0_ft.NUM_CLASSES
IGNORE_INDEX = _a0_ft.IGNORE_INDEX
OUTPUT_STRIDE = _a0_ft.OUTPUT_STRIDE
FINETUNE_MAX_STEPS = _a0_ft.FINETUNE_MAX_STEPS
ACCEPTED_PROBE_ARTIFACT_TYPES = _a0_ft.ACCEPTED_PROBE_ARTIFACT_TYPES

parse_args = _a0_ft.parse_args
ft_paths = _a0_ft.ft_paths
default_probe_checkpoint = _a0_ft.default_probe_checkpoint
load_probe_as_finetune_start = _a0_ft.load_probe_as_finetune_start
build_finetune_checkpoint = _a0_ft.build_finetune_checkpoint
load_finetune_model = _a0_ft.load_finetune_model
run_training = _a0_ft.run_training


def main() -> None:
    _a0_ft.main()


if __name__ == "__main__":
    _a0_ft.torch.multiprocessing.freeze_support()
    main()
