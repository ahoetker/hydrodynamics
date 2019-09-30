"""Make CSV
This module digests LabQuest-format .txt files and outputs CSVs

The regular expressions and column names in this module correspond to the
Continuous Reactor Hydrodynamics experiment in CHE 451. Modify as needed.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Pattern for conductivity data in this experiment
r = re.compile(r"(\d+)\t(.+)\t(.+)\t(.+)\t")


def labquest_txt_to_df(txtfile: Path) -> pd.DataFrame:
    with txtfile.open("r") as f:
        text = f.read()
    time = [np.float64(m.group(1)) for m in re.finditer(r, text)]
    cond1 = [np.float64(m.group(2)) for m in re.finditer(r, text)]
    cond2 = [np.float64(m.group(3)) for m in re.finditer(r, text)]
    cond3 = [np.float64(m.group(4)) for m in re.finditer(r, text)]
    df = pd.DataFrame(
        {
            "Time (s)": time,
            "CH1 Conductivity (muS/cm)": cond1,
            "CH2 Conductivity (muS/cm)": cond2,
            "CH3 Conductivity (muS/cm)": cond3,
        }
    )
    return df


def make_all_csvs() -> None:
    data_dir = Path("data")
    for txtfile in data_dir.glob("**/*.txt"):
        df = labquest_txt_to_df(txtfile)
        outfile = txtfile.with_suffix(".csv")
        df.to_csv(outfile, index=False)
        print("Wrote data to {}".format(outfile))


if __name__ == "__main__":
    make_all_csvs()
