from pathlib import Path

import numpy as np
import pandas as pd
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


class Reactor:
    def __init__(self, design: str):
        self.design = design


class PFR(Reactor):
    def __init__(self, length, OD, wall_thickness, label):
        super().__init__("PFR")
        self.length = length
        self.ID = OD - 2 * wall_thickness
        self.area = np.pi * (self.ID / 2) ** 2
        self.volume = self.area * self.length
        self.label = label


class CSTR(Reactor):
    def __init__(self, volume, geometry):
        super().__init__("CSTR")
        self.volume = volume
        self.geometry = geometry


class CSTRSeries(Reactor):
    def __init__(self, volume):
        super().__init__("CSTR Series")
        self.volume = volume


class Trial:
    def __init__(self, csv: Path, reactor: Reactor, flowrate, agitation: int = None):
        self.csv = csv
        self.reactor = reactor
        self.flowrate = flowrate
        self.agitation = agitation
        with csv.open("r") as csvfile:
            self.data = pd.read_csv(csvfile)
        self.reynolds = None
        self.baseline = None

    def set_baseline(self, baseline):
        self.baseline = baseline

    def set_reynolds(self, reynolds):
        self.reynolds = reynolds

    def timeseries(self) -> np.array:
        return self.data["Time (s)"]

    def corrected_cond(self) -> np.array:
        """ get largest-range conductivity series
        """
        cond_cols = [c for c in self.data.columns if "Cond" in c]
        ranges = [self.data[c].max() - self.data[c].min() for c in cond_cols]
        cond = self.data[cond_cols[ranges.index(max(ranges))]]
        return cond - self.baseline
