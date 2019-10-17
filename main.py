# %% Setup
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pint
import numpy as np
from collections import OrderedDict
from scipy import stats
from typing import List, Tuple, Iterable
from dataclasses import Impeller, PFR, CSTR, CSTRSeries, Trial

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


def impeller_reynolds(rho, N, D, mu):
    """Dimensionless Reynolds number for CSTR
    """
    rho_SI = rho.to("kg per cubic meter")
    N_SI = N.to("rpm")
    D_SI = D.to("meter")
    mu_SI = mu.to("kg per meter per second")
    N_Re = ((rho_SI * N_SI * D_SI**2) / mu_SI).to("dimensionless")
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


def get_zero(a: np.array) -> Iterable[int]:
    for i in range(len(a) - 1):
        if a[i] * a[i + 1] <= 0:
            yield i + 1


def get_inflection(a: np.array) -> Iterable[int]:
    for i in range(len(a) - 1):
        lslope = a[i] - a[i - 1]
        rslope = a[i + 1] - a[i]
        if lslope * rslope <= 0:
            yield i


def select_peak(t: np.array, y: np.array, n_zeros: int) -> Tuple[np.array, np.array]:
    global_max_ix = np.argmax(y)
    right = y[global_max_ix + 1:]
    left = y[global_max_ix - 1::-1]
    rzeros_ix = [z + global_max_ix for z in get_inflection(right)][:n_zeros]
    lzeros_ix = [global_max_ix - z for z in get_inflection(left)][:n_zeros]
    if len(rzeros_ix) == 0:
        return select_peak(t, y[:-1], n_zeros)
    elif len(lzeros_ix) == 0:
        return select_peak(t, y[1:], n_zeros)
    else:
        y_selected = y[lzeros_ix[-1]:rzeros_ix[-1] + 1]
        t_selected = np.linspace(lzeros_ix[-1], rzeros_ix[-1] + 1, len(t))
        return t[lzeros_ix[-1]:rzeros_ix[-1] + 1], y[lzeros_ix[-1]:rzeros_ix[-1] + 1]


def rtd(t: np.array, cond: np.array) -> np.array:
    """ This is the 'E' function
    """
    cond = cond.to("microsiemens / cm").magnitude
    E = cond / np.trapz(cond, t)
    return Q_(E, "1 / second")


def cum_dist(t: np.array, cond: np.array) -> np.array:
    """ This is the 'F' function
    """
    E = rtd(t, cond)
    F = np.cumsum(E)
    return Q_(F, "dimensionless")


def t_mean(t: np.array, cond: np.array) -> np.array:
    E = rtd(t, cond)
    t_m = np.trapz(t * E, t)
    return t_m


def spike_variance(t: np.array, cond: np.array) -> np.array:
    E = rtd(t, cond)
    t_m = np.trapz(t * E, t)
    variance = np.trapz((t - t_m)**2 * E, t)
    return variance


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

# Impeller specifications
imp_1 = Impeller("1", Q_("100 mm"))
imp_2 = Impeller("2", Q_("50 mm"))

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
            if geometry == "1":
                geometry = imp_1
            else:
                geometry = imp_2
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

inflection_step = 4
polynomial_degree = 8
for trial in pfr_trials:
    velocity = trial.flowrate / trial.reactor.area
    N_Re = reynolds(rho, trial.reactor.ID, velocity, mu)
    trial.set_reynolds(N_Re)
    trial.set_baseline(980)
    t = np.array(trial.timeseries())
    y = np.array(trial.corrected_cond())
    ts, ys = smooth(t, y, 11, "hamming")
    t_selected, y_selected = select_peak(ts, ys, inflection_step)
    trial.set_peak_selection(t_selected, y_selected)

cstr_trials = list(filter(lambda t: t.reactor.design == "CSTR", trials))

inflection_step = 6
polynomial_degree = 8
for trial in cstr_trials:
    N = trial.agitation
    D = trial.reactor.geometry.diameter
    N_Re = impeller_reynolds(rho, N, D, mu)
    trial.set_reynolds(N_Re)
    if trial.data["CH1 Conductivity (muS/cm)"].min() < 1000:
        trial.set_baseline(998)
    else:
        trial.set_baseline(1145)
    t = np.array(trial.timeseries())
    y = np.array(trial.corrected_cond())
    ts, ys = smooth(t, y, 11, "hamming")
    t_selected, y_selected = select_peak(ts, ys, inflection_step)
    trial.set_peak_selection(t_selected, y_selected)


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
plt.savefig(Path(figures_dir / "pfr_flowrate_reynolds.png"), bbox_inches="tight")
plt.show()

# TODO Plot: Conductivity vs time

## Conductivity for PFRs, all trials, no smoothing
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"Conductivity ($\mu$S / cm)")
for trial in pfr_trials:
    y = trial.corrected_cond()
    ax1.plot(y)
plt.savefig(str(figures_dir / "pfr_conductivity_all_nosmooth.png"), bbox_inches="tight")
plt.show()

## Conductivity for PFRs, all trials
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Conductivity ($\mu$S / cm)")
for trial in pfr_trials:
    t = np.array(trial.timeseries())
    y = trial.corrected_cond()
    ts, ys = smooth(t, y, 11, "hamming")
    ax1.plot(ts, ys)
plt.savefig(str(figures_dir / "pfr_conductivity_all.png"), bbox_inches="tight")
plt.show()

## Conductivity for PFRs, color coded for length
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"Conductivity ($\mu$S / cm)")
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
plt.savefig(str(figures_dir / "pfr_conductivity_by_length.png"), bbox_inches="tight")
plt.show()

## Conductivity for PFRs, color coded for flowrate
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"Conductivity ($\mu$S / cm)")
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
plt.savefig(str(figures_dir / "pfr_conductivity_by_flowrate.png"), bbox_inches="tight")
plt.show()

# TODO Plot: E(t) vs time

## E(t) for PFRs, all trials
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("E(t) (1/s)")
for trial in pfr_trials:
    t = trial.t_selected
    cond = Q_(trial.cond_selected, "microsiemens / cm")
    E = rtd(t, cond)
    ax1.plot(t, E)
plt.savefig(str(figures_dir / "E_pfr_all.png"), bbox_inches="tight")
plt.show()

## E(t) for PFRs, color coded for length
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("E(t) (1/s)")
for trial in pfr_trials:
    t = trial.t_selected
    cond = Q_(trial.cond_selected, "microsiemens / cm")
    E = rtd(t, cond)
    if trial.reactor.label == "short PFR":
        ax1.plot(t, E, label="short PFR", color=colors[0]["color"])
    elif trial.reactor.label == "long PFR":
        ax1.plot(t, E, label="long PFR", color=colors[1]["color"])
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
plt.savefig(str(figures_dir / "E_pfr_by_length.png"), bbox_inches="tight")
plt.show()

## E(t) for PFRs, color coded for flowrate
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("E(t) (1/s)")
for trial in pfr_trials:
    t = trial.t_selected
    cond = Q_(trial.cond_selected, "microsiemens / cm")
    E = rtd(t, cond)
    if trial.flowrate == Q_("400 mL/min"):
        ax1.plot(t, E, label="400 mL/min", color=colors[0]["color"])
    elif trial.flowrate == Q_("800 mL/min"):
        ax1.plot(t, E, label="800 mL/min", color=colors[1]["color"])
    elif trial.flowrate == Q_("1200 mL/min"):
        ax1.plot(t, E, label="1200 mL/min", color=colors[2]["color"])
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
plt.savefig(str(figures_dir / "E_pfr_by_flowrate.png"), bbox_inches="tight")
plt.show()

# TODO Plot: F(t) vs time
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("F(t)")
for trial in pfr_trials:
    t = trial.t_selected
    cond = Q_(trial.cond_selected, "microsiemens / cm")
    F = cum_dist(t, cond)
    ax1.plot(t, F)
plt.savefig(str(figures_dir / "F_pfr_all.png"), bbox_inches="tight")
plt.show()

# %%
# TODO Plot: t*E(t) vs time
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"$tE(t)$")
trial = pfr_trials[7]
t = trial.t_selected
cond = Q_(trial.cond_selected, "microsiemens / cm")
E = cum_dist(t, cond)
ax1.plot(t, t * E)
plt.show()


# %% Plot: E(t-t_m)**2 vs t
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"$E(t-t_m)^2$")
trial = pfr_trials[2]
t = trial.t_selected
cond = Q_(trial.cond_selected, "microsiemens / cm")
E = cum_dist(t, cond)
t_m = t_mean(t, cond)
ax1.plot(t, ((t - t_m)**2 * E))
plt.show()


# %% Plot: variance vs reynolds number for PFRs
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Reynolds Number")
ax1.set_ylabel(r"Variance ($\sigma^2$)")
N_Re_short = []
N_Re_long = []
variance_short = []
variance_long = []
for trial in pfr_trials:
    t = np.array(trial.t_selected)
    cond = Q_(trial.cond_selected, "microsiemens / cm")
    trial_variance = spike_variance(t, cond)
    if trial.reactor.label == "short PFR":
        variance_short.append(trial_variance)
        N_Re_short.append(trial.reynolds.magnitude)
    elif trial.reactor.label == "long PFR":
        variance_long.append(trial_variance)
        N_Re_long.append(trial.reynolds.magnitude)
ax1.scatter(N_Re_short, variance_short, c="k", marker="x", label="short PFR")
ax1.scatter(N_Re_long, variance_long, c="k", marker="o", label="long PFR")
ax1.axvspan(0, 2100, alpha=0.1, color="blue")
ax1.axvspan(2100, 4000, alpha=0.1, color="green")
ax1.axvspan(4000, 7500, alpha=0.1, color="yellow")
ax1.legend()
plt.savefig(str(figures_dir / "variance_reynolds_pfr.png"), bbox_inches="tight")
plt.show()

# %% "Data story" for PFR
trial = pfr_trials[7]
fig = plt.figure(figsize=(7.5, 4.5), dpi=300, )
ax1 = fig.add_subplot(221)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"$E(t)$ (1/s)")
t = np.array(trial.t_selected)
cond = Q_(trial.cond_selected, "microsiemens / cm")
E = rtd(t, cond)
ax1.plot(t, E.magnitude)

ax2 = fig.add_subplot(222)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel(r"$F(t)$")
F = cum_dist(t, cond)
ax2.plot(t, F.magnitude)

ax3 = fig.add_subplot(223)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel(r"$tE(t)$")
ax3.plot(t, t * E.magnitude)

ax4 = fig.add_subplot(224)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel(r"$(t-t_m)^2E(t)$")
t_m = t_mean(t, cond)
ax4.plot(t, ((t - t_m)**2 * E))

plt.tight_layout()
plt.savefig(str(figures_dir / "pfr_data_story.png"), bbox_inches="tight")
plt.show()


# %% Demo plot of smoothing
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Conductivity ($\mu$S / cm)")
trial = pfr_trials[5]
x = np.array(trial.timeseries())
y = trial.corrected_cond()
xs, ys = smooth(x, y, 11, "hamming")
ax1.plot(x, y, "-k", label="Instrument", linewidth=0.6)
ax1.plot(xs, ys, "-b", label="Convolved")
ax1.plot(trial.t_selected, trial.cond_selected, "-r", label="Selected")
ax1.legend()
plt.savefig(str(figures_dir / "smoothing_demo.png"), bbox_inches="tight")
plt.show()

# %% CSTR Plots
## Conductivity for CSTRs, all trials, no smoothing
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"Conductivity ($\mu$S / cm)")
for trial in cstr_trials:
    y = trial.corrected_cond()
    ax1.plot(y)
plt.savefig(str(figures_dir / "cstr_conductivity_all_nosmooth.png"), bbox_inches="tight")
plt.show()

## Conductivity for CSTRs, all trials
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Conductivity ($\mu$S / cm)")
for trial in cstr_trials:
    t = np.array(trial.timeseries())
    y = trial.corrected_cond()
    ts, ys = smooth(t, y, 11, "hamming")
    ax1.plot(ts, ys)
plt.savefig(str(figures_dir / "cstr_conductivity_all.png"), bbox_inches="tight")
plt.show()

## Conductivity for CSTRs, color coded for flowrate
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"Conductivity ($\mu$S / cm)")
for trial in cstr_trials:
    t = np.array(trial.timeseries())
    y = trial.corrected_cond()
    ts, ys = smooth(t, y, 11, "hamming")
    if trial.flowrate == Q_("600 mL/min"):
        ax1.plot(ts, ys, label="600 mL/min", color=colors[0]["color"])
    elif trial.flowrate == Q_("1200 mL/min"):
        ax1.plot(ts, ys, label="1200 mL/min", color=colors[1]["color"])
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
plt.savefig(str(figures_dir / "cstr_conductivity_by_flowrate.png"), bbox_inches="tight")
plt.show()


# %% Plot: variance vs reynolds number for CSTRs
fig = plt.figure(figsize=(5, 3), dpi=300)
ax1 = fig.add_subplot(111)
ax1.set_xlabel("Reynolds Number")
ax1.set_ylabel(r"Variance ($\sigma^2$)")
N_Re_1 = []
N_Re_2 = []
variance_1 = []
variance_2 = []
for trial in cstr_trials:
    t = np.array(trial.t_selected)
    cond = Q_(trial.cond_selected, "microsiemens / cm")
    trial_variance = spike_variance(t, cond)
    if trial.reactor.geometry.label == "1":
        variance_1.append(trial_variance)
        N_Re_1.append(trial.reynolds.magnitude)
    elif trial.reactor.geometry.label == "2":
        variance_2.append(trial_variance)
        N_Re_2.append(trial.reynolds.magnitude)
ax1.scatter(N_Re_1, variance_1, c="k", marker="x", label="100mm Impeller")
ax1.scatter(N_Re_2, variance_2, c="k", marker="o", label="50mm Impeller")
ax1.legend()
plt.savefig(str(figures_dir / "variance_reynolds_cstr.png"), bbox_inches="tight")
plt.show()

# %% "Data story" for CSTR
trial = cstr_trials[1]
fig = plt.figure(figsize=(7.5, 4.5), dpi=300, )
ax1 = fig.add_subplot(221)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"$E(t)$ (1/s)")
t = np.array(trial.t_selected)
cond = Q_(trial.cond_selected, "microsiemens / cm")
E = rtd(t, cond)
ax1.plot(t, E.magnitude)

ax2 = fig.add_subplot(222)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel(r"$F(t)$")
F = cum_dist(t, cond)
ax2.plot(t, F.magnitude)

ax3 = fig.add_subplot(223)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel(r"$tE(t)$")
ax3.plot(t, t * E.magnitude)

ax4 = fig.add_subplot(224)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel(r"$(t-t_m)^2E(t)$")
t_m = t_mean(t, cond)
ax4.plot(t, ((t - t_m)**2 * E))

plt.tight_layout()
plt.savefig(str(figures_dir / "cstr_data_story.png"), bbox_inches="tight")
plt.show()
# %% Tables

# %% Output spreadsheets
