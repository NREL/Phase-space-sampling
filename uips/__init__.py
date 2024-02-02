"""Phase space sampling"""

import os

from uips.version import __version__

UIPS_DIR = os.path.dirname(os.path.realpath(__file__))
UIPS_INPUT_DIR = os.path.join(UIPS_DIR, "inputs")
UIPS_DATA_DIR = os.path.join(UIPS_DIR, "../data")
UIPS_DOC_DIR = os.path.join(UIPS_DIR, "../documentation")
UIPS_ML_DIR = os.path.join(UIPS_DIR, "../data-efficientML")
