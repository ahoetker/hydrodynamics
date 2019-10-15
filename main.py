# The usual suspects
import re
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pint


from dataclasses import PFR, CSTR, CSTRSeries, Trial


# Setup
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
ureg.setup_matplotlib()
figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"


def reynolds(rho, D, v, mu):
    """Dimensionless Reynolds number
    """
    rho_SI = rho.to("kg per cubic meter")
    D_SI = D.to("meter")
    v_SI = v.to("meter per second")
    mu_SI = mu.to("kg per meter per second")
    N_Re = ((rho_SI * D_SI * v_SI) / mu_SI).to("dimensionless")
    return N_Re


# Constant specifications
rho = Q_("997 kg per cubic meter")
mu = Q_("0.89 mPa * s")

# Reactor specifications
short_pfr = PFR(
    length=Q_("39 feet"), OD=Q_("0.375 inch"), wall_thickness=Q_("0.062 inch")
)
long_pfr = PFR(
    length=Q_("100 feet"), OD=Q_("0.25 inch"), wall_thickness=Q_("0.047 inch")
)
cstr_series = CSTRSeries(volume=Q_("3 liter"))

# Instantiate list of Trial objects from csv files
trials = []
data_dir = Path("data")
for csvfile in data_dir.glob("**/*.csv"):
    if "cstr" in csvfile.name:
        r = re.compile(r"cstr_(\d+)_(\d+).+")
        m = re.match(r, csvfile.name)
        flowrate = Q_(int(m.group(1)), "mL/min")
        agitation = Q_(int(m.group(2)), "rpm")
        if "series" in csvfile.name:
            trial = Trial(csvfile, cstr_series, flowrate, agitation)
            trials.append(trial)
        else:
            r = re.compile(r"cstr_(\d+)_(\d+)_(\d+).+")
            m = re.match(r, csvfile.name)
            geometry = m.group(3)
            reactor = CSTR(Q_("3 liter"), geometry)
            trial = Trial(csvfile, reactor, flowrate, agitation)
            trials.append(trial)
    elif "pfr" in csvfile.name:
        r = re.compile(r"pfr_(\d+).+")
        m = re.match(r, csvfile.name)
        flowrate = Q_(int(m.group(1)), "mL/min")
        if "025" in csvfile.name:
            trial = Trial(csvfile, long_pfr, flowrate)
            trials.append(trial)
        elif "0375" in csvfile.name:
            trial = Trial(csvfile, short_pfr, flowrate)
            trials.append(trial)

pfr_trials = list(filter(lambda t: t.reactor.design == "PFR", trials))

for trial in pfr_trials:
    velocity = trial.flowrate / trial.reactor.area
    N_Re = reynolds(rho, trial.reactor.ID, velocity, mu)
    trial.set_reynolds(N_Re)

# PFR conductivity plots
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_xlabel("Time (s)")
# ax1.set_ylabel("Conductivity (mSiemens/cm)")
# for trial in pfr_trials:
#     flowrate = trial.flowrate.magnitude
#     diameter = trial.reactor.ID.magnitude
#     label_text = f"Flowrate: {flowrate} mL/min, Diameter: {diameter} in."
#     ax1.plot(
#         trial.data["Time (s)"],
#         trial.data["CH3 Conductivity (muS/cm)"],
#         label=label_text,
#     )
# ax1.legend()
# plt.show()

# PFR Vol-Flowrate-Reynolds plot
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
short = [trial for trial in pfr_trials if trial.reactor is short_pfr]
long = [trial for trial in pfr_trials if trial.reactor is long_pfr]
for split_trials in [short, long]:
    x = [trial.flowrate.to("mL/min").magnitude for trial in split_trials]
    y = [trial.reynolds.magnitude for trial in split_trials]
    ax1.plot(x, y, "o", markerfacecolor="w", label=f"{split_trials[0].reactor.length}")
ax1.set_xlabel("Flowrate (mL/min)")
ax1.set_ylabel("Reynolds Number")
ax1.legend()
plt.savefig(Path(figures_dir / "pfr_flowrate_reynolds.pdf"), bbox_inches="tight")
plt.show()

# Preliminary conductivity/cumsum plot for sanity check
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
pfr_spike = pfr_trials[0]
t = pfr_spike.data["Time (s)"]

# TODO: Plot mean residence time vs. Reynolds number
