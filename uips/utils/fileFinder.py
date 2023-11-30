import os
import sys

import uips.utils.parallel as par
from uips import PSS_DATA_DIR, PSS_INPUT_DIR


def find_input(inpt_file):
    if not os.path.isfile(inpt_file):
        new_inpt_file = os.path.join(
            PSS_INPUT_DIR, os.path.split(inpt_file)[-1]
        )
        par.printRoot(
            f"WARNING: {inpt_file} not found trying {new_inpt_file} ..."
        )
        if not os.path.isfile(new_inpt_file):
            par.printRoot(
                f"ERROR: could not open data {inpt_file} or {new_inpt_file}"
            )
            sys.exit()
            return None
        else:
            return new_inpt_file
    else:
        return inpt_file


def find_data(data_file):
    if not os.path.isfile(data_file):
        new_data_file = os.path.join(
            PSS_DATA_DIR, os.path.split(data_file)[-1]
        )
        par.printRoot(
            f"WARNING: {data_file} not found trying {new_data_file} ..."
        )
        if not os.path.isfile(new_data_file):
            par.printRoot(
                f"ERROR: could not open data {data_file} or {new_data_file}"
            )
            sys.exit()
            return None
        else:
            return new_data_file
    else:
        return data_file
