# %% Setup
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pint
import numpy as np
import pandas as pd
from collections import OrderedDict
from dataclasses import PFR, CSTR, CSTRSeries, Trial

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
ureg.setup_matplotlib()
figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)
colors = [k for k in mpl.rcParams["axes.prop_cycle"]]
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"


# %% Functions


def reynolds(rho, D, v, mu):
    """ Dimensionless Reynolds number
    """
    rho_SI = rho.to("kg per cubic meter")
    D_SI = D.to("meter")
    v_SI = v.to("meter per second")
    mu_SI = mu.to("kg per meter per second")
    N_Re = ((rho_SI * D_SI * v_SI) / mu_SI).to("dimensionless")
    return N_Re


def smooth(t, x, window_len=11, window="hanning"):
    """ Apply smoothing to noisy data.
    See: https://en.wikipedia.org/wiki/Window_function#Cosine-sum_windows

    CURRENTLY IGNORING window_len
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1-d arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    lininterp = lambda x1, x2, y1, y2, x: y1 + (x - x1) / (x2 - x1) * (y2 - y1)
    window_len = int(np.ceil(lininterp(80, 400, 6, 11, x.size)))

    if window_len < 3:
        return t, x

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    if window == "flat":
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    ts = np.linspace(0, t[-1], len(y))
    return ts, y


def rtd(conductivity: np.array):
    """ This is the 'E' function
    """
    cumsum = Q_(sum(conductivity.magnitude), "microsiemens / cm * second")
    E = conductivity / cumsum
    return E.to("1 / second")


def cum_dist(conductivity: np.array):
    """ This is the 'F' function
    """
    E = rtd(conductivity)
    F = Q_(np.cumsum(E).magnitude, "dimensionless")
    return F


# %% Specs
rho = Q_("997 kg per cubic meter")
mu = Q_("0.89 mPa * s")

# Reactor specifications
short_pfr = PFR(
    length=Q_("39 feet"),
    OD=Q_("0.375 inch"),
    wall_thickness=Q_("0.062 inch"),
    label="short PFR",
)
long_pfr = PFR(
    length=Q_("100 feet"),
    OD=Q_("0.25 inch"),
    wall_thickness=Q_("0.047 inch"),
    label="long PFR",
)
cstr_series = CSTRSeries(volume=Q_("3 liter"))

# %% Runtime data manipulation
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
    trial.set_baseline(980)


# %% Plots
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

# TODO Plot: Conductivity vs time

## Conductivity for PFRs, all trials, no smoothing
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Conductivity (muS / cm)")
for trial in pfr_trials:
    y = trial.corrected_cond()
    ax1.plot(y)
plt.savefig(str(figures_dir / "pfr_conductivity_all_nosmooth.pdf"), bbox_inches="tight")
plt.show()

## Conductivity for PFRs, all trials
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Conductivity (muS / cm)")
for trial in pfr_trials:
    t = np.array(trial.timeseries())
    y = trial.corrected_cond()
    ts, ys = smooth(t, y, 11, "hamming")
    ax1.plot(ts, ys)
plt.savefig(str(figures_dir / "pfr_conductivity_all.pdf"), bbox_inches="tight")
plt.show()

## Conductivity for PFRs, color coded for length
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Conductivity (muS / cm)")
for trial in pfr_trials:
    t = np.array(trial.timeseries())
    y = trial.corrected_cond()
    ts, ys = smooth(t, y, 11, "hamming")
    if trial.reactor.label == "short PFR":
        ax1.plot(ts, ys, label="short PFR", color=colors[0]["color"])
    elif trial.reactor.label == "long PFR":
        ax1.plot(ts, ys, label="long PFR", color=colors[1]["color"])
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
plt.savefig(str(figures_dir / "pfr_conductivity_by_length.pdf"), bbox_inches="tight")
plt.show()

## Conductivity for PFRs, color coded for flowrate
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Conductivity (muS / cm)")
for trial in pfr_trials:
    t = np.array(trial.timeseries())
    y = trial.corrected_cond()
    ts, ys = smooth(t, y, 11, "hamming")
    if trial.flowrate == Q_("400 mL/min"):
        ax1.plot(ts, ys, label="400 mL/min", color=colors[0]["color"])
    elif trial.flowrate == Q_("800 mL/min"):
        ax1.plot(ts, ys, label="800 mL/min", color=colors[1]["color"])
    elif trial.flowrate == Q_("1200 mL/min"):
        ax1.plot(ts, ys, label="1200 mL/min", color=colors[2]["color"])
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
plt.savefig(str(figures_dir / "pfr_conductivity_by_flowrate.pdf"), bbox_inches="tight")
plt.show()

# TODO Plot: E(t) vs time

## E(t) for PFRs, all trials
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("E(t) (1/s)")
for trial in pfr_trials:
    t = np.array(trial.timeseries())
    cond = trial.corrected_cond()
    ts, conds = smooth(t, cond, 11, "hamming")
    conds = Q_(conds, "microsiemens / cm")
    ys = rtd(conds).magnitude
    ax1.plot(ts, ys)
plt.savefig(str(figures_dir / "E_pfr_all.pdf"), bbox_inches="tight")
plt.show()

## E(t) for PFRs, color coded for length
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("E(t) (1/s)")
for trial in pfr_trials:
    t = np.array(trial.timeseries())
    cond = trial.corrected_cond()
    ts, conds = smooth(t, cond, 11, "hamming")
    conds = Q_(conds, "microsiemens / cm")
    ys = rtd(conds).magnitude
    if trial.reactor.label == "short PFR":
        ax1.plot(ts, ys, label="short PFR", color=colors[0]["color"])
    elif trial.reactor.label == "long PFR":
        ax1.plot(ts, ys, label="long PFR", color=colors[1]["color"])
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
plt.savefig(str(figures_dir / "E_pfr_by_length.pdf"), bbox_inches="tight")
plt.show()

## E(t) for PFRs, color coded for flowrate
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("E(t) (1/s)")
for trial in pfr_trials:
    t = np.array(trial.timeseries())
    cond = trial.corrected_cond()
    ts, conds = smooth(t, cond, 11, "hamming")
    conds = Q_(conds, "microsiemens / cm")
    ys = rtd(conds).magnitude
    if trial.flowrate == Q_("400 mL/min"):
        ax1.plot(ts, ys, label="400 mL/min", color=colors[0]["color"])
    elif trial.flowrate == Q_("800 mL/min"):
        ax1.plot(ts, ys, label="800 mL/min", color=colors[1]["color"])
    elif trial.flowrate == Q_("1200 mL/min"):
        ax1.plot(ts, ys, label="1200 mL/min", color=colors[2]["color"])
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
plt.savefig(str(figures_dir / "E_pfr_by_flowrate.pdf"), bbox_inches="tight")
plt.show()

# TODO Plot: F(t) vs time
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("F(t)")
for trial in pfr_trials:
    t = np.array(trial.timeseries())
    cond = trial.corrected_cond()
    ts, conds = smooth(t, cond, 11, "hamming")
    conds = Q_(conds, "microsiemens / cm")
    ys = cum_dist(conds).magnitude
    ax1.plot(ts, ys)
plt.savefig(str(figures_dir / "F_pfr_all.pdf"), bbox_inches="tight")
plt.show()
# TODO Plot: t*E(t) vs time

# %%
# TODO Plot: E*(t-tm)^2 (plot whose integral represents variance)

## Demo of smoothing effect on variance plot
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"$E(t-t_m)^2$")
trial = pfr_trials[4]
t = np.array(trial.timeseries())
cond = Q_(trial.corrected_cond(), "microsiemens/cm")
y = rtd(cond) * (t - t.mean()) ** 2
ax1.plot(t, y.magnitude, linewidth=0.8, label="Raw conductivity")
ts, conds = smooth(t, trial.corrected_cond(), 11, "hamming")
conds = Q_(conds, "microsiemens/cm")
ys = rtd(conds) * (ts - ts.mean()) ** 2
ax1.plot(ts, ys.magnitude, "k", label="Convolved conductivity")
ax1.legend()
plt.savefig(str(figures_dir / "smoothing_result_demo.pdf"), bbox_inches="tight")
plt.show()

# %% Demo plot of smoothing
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Conductivity (muS / cm)")
trial = pfr_trials[4]
x = np.array(trial.timeseries())
y = trial.corrected_cond()
xs, ys = smooth(x, y, 11, "hamming")
ax1.plot(x, y, label="Raw", linewidth=0.8)
ax1.plot(xs, ys, "-k", label="Convolved")
ax1.legend()
plt.savefig(str(figures_dir / "smoothing_demo.pdf"), bbox_inches="tight")
plt.show()

# %% testing plot of t*E(t)
## E(t) for PFRs, all trials
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("E(t) (1/s)")
for trial in pfr_trials:
    t = np.array(trial.timeseries())
    cond = Q_(trial.corrected_cond(), "microsiemens / cm")
    y = rtd(cond).magnitude * np.arange(0, len(cond))
    ts, conds = smooth(t, cond.magnitude, 11, "hamming")
    conds = Q_(conds, "microsiemens / cm")
    ys = rtd(conds).magnitude * np.arange(0, len(conds))
    ax1.plot(ts, ys)

plt.show()


# %% Tables

# %% Output spreadsheets
