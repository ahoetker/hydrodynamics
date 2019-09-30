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

    def __init__(self, length, OD, wall_thickness):
        super().__init__("PFR")
        self.length = length
        self.ID = OD - 2 * wall_thickness
        self.area = np.pi * (self.ID / 2) ** 2
        self.volume = self.area * self.length


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

    def set_reynolds(self, reynolds):
        self.reynolds = reynolds
