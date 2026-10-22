
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
"""Рисунки к черновику и проверка чисел.

Запуск из этой папки: ../py make_figures.py
"""
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref_milp

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "output"

# Обычные сутки, G2: час 0..23, солнечный горб днём, вечерний пик −1.
DAY = [
    -0.25, -0.20, -0.15, -0.15, -0.15, -0.20, -0.45, -0.60,
    -0.40, 0.20, 0.70, 1.00, 1.10, 1.00, 0.60, 0.20,
    -0.30, -0.80, -1.00, -1.00, -0.80, -0.50, -0.35, -0.30,
]

# Верхние концы биссекции G2 для бака E=3 на этих сутках.
G2_TANK = {0.005: 0.003312, 0.01: 0.006687, 0.02: 0.011034}


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.titlesize": 11,
        "figure.dpi": 120,
    })


def comma_axis(ax):
    """Десятичная запятая на осях."""
    from matplotlib.ticker import FuncFormatter

    def fmt(x, _pos):
        text = f"{x:g}"
        return text.replace(".", ",")

    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))


def runs_of_positive(values, eps=1e-12):
    runs = []
    start = None
    for i, v in enumerate(values):
        if v > eps:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def simulate(c, d, eta, delta, e_cap):
    """Прямой прогон циклической динамики. Возвращает запас на конец каждого часа и e_0."""
    gamma = 1.0 - delta
    T = len(c)
    # e_0 подбирается замыканием: при известных c, d запас — аффинная функция e_0.
    # e_t = γ e_{t-1} + η c_t − d_t, e_T = e_0.
    acc = 0.0
    weight = 1.0
    for t in range(T):
        acc = gamma * acc + eta * c[t] - d[t]
        weight *= gamma
    # e_T = weight * e_0 + acc, и e_T = e_0.
    denom = 1.0 - weight
    if abs(denom) < 1e-15:
        if abs(acc) > 1e-9:
            raise RuntimeError("цикл без саморазряда не замкнулся")
        e0 = 0.0
    else:
        e0 = acc / denom
    e = []
    prev = e0
    for t in range(T):
        prev = gamma * prev + eta * c[t] - d[t]
        if prev < -1e-9 or prev > e_cap + 1e-9:
            raise RuntimeError("запас вне [0, E] в часе %d: %s" % (t + 1, prev))
        e.append(prev)
    if abs(e[-1] - e0) > 1e-8:
        raise RuntimeError("цикл не сошёлся")
    return e0, e


def check_quarter():
    """Пример с четвертью: три режима и порог 1/4 рационально."""
    x = [Fraction(1), Fraction(-1, 2), Fraction(-1, 2), Fraction(-1), Fraction(-1), Fraction(-1)]
    r = Fraction(1, 4)

    def mode(charge, discharge, reserve):
        deficit = [max(-xi - reserve, 0) for xi in x]
        ell = [deficit[t] - discharge[t] for t in range(6)]
        if any(v < 0 for v in ell):
            raise RuntimeError("разряд больше дефицита")
        if sum(discharge) > 1:
            raise RuntimeError("разряд больше ёмкости")
        return deficit, discharge, ell

    h_def, h_d, h_ell = mode(
        [Fraction(1), 0, 0, 0, 0, 0],
        [0, Fraction(1, 2), Fraction(1, 2), 0, 0, 0],
        Fraction(0),
    )
    d_def, d_d, d_ell = mode(
        [Fraction(1), 0, 0, 0, 0, 0],
        [0, 0, 0, Fraction(1), 0, 0],
        Fraction(0),
    )
    j_def, j_d, j_ell = mode(
        [Fraction(1), 0, 0, 0, 0, 0],
        [0, Fraction(1, 4), 0, Fraction(3, 4), 0, 0],
        r,
    )
    for name, c, dch in (
        ("H", [1.0, 0, 0, 0, 0, 0], [0, 0.5, 0.5, 0, 0, 0]),
        ("D", [1.0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0]),
        ("J", [1.0, 0, 0, 0, 0, 0], [0, 0.25, 0, 0.75, 0, 0]),
    ):
        simulate(c, dch, 1.0, 0.0, 1.0)

    def n_pos(ell):
        return sum(1 for v in ell if v > 0)

    def max_run(ell):
        runs = runs_of_positive([float(v) for v in ell])
        return max((b - a + 1) for a, b in runs)

    if n_pos(h_ell) != 3 or max_run(h_ell) != 3:
        raise RuntimeError("режим H: ожидались 3 часа одной серией")
    if n_pos(d_ell) != 4 or max_run(d_ell) != 2:
        raise RuntimeError("режим D: ожидались 4 часа, серия 2")
    if n_pos(j_ell) != 3 or max_run(j_ell) != 2:
        raise RuntimeError("совместный режим: ожидались H=3 и D=2")
    # Самый дешёвый совместный набор: один дорогой час и один дешёвый.
    # Два дешёвых часа не рвут серию 4–6. Энергия 3/2−2r > 1 при r < 1/4.
    if Fraction(3, 2) - 2 * r != 1:
        raise RuntimeError("на пороге энергия должна равняться 1")
    print("quarter ok", "H", [str(v) for v in h_ell], "D", [str(v) for v in d_ell], "J", [str(v) for v in j_ell])
    return {
        "H": (h_def, h_d, h_ell),
        "D": (d_def, d_d, d_ell),
        "J": (j_def, j_d, j_ell),
    }


def shore_gamma(delta, L):
    """Γ при E = 1. delta и L могут быть Fraction."""
    gamma = 1 - delta
    gL = gamma ** L
    return gamma * (1 - gL) / (L + gL ** (-1))


def shore_ratio_float(delta, L):
    """Γ/(δE) в вещественной арифметике. E = 1."""
    gamma = 1.0 - delta
    # γ^{−L} = exp(−L ln γ) не переполняется раньше, чем отношение уже нулевое.
    log_inv = -L * np.log(gamma)
    if log_inv > 700:
        return 0.0
    g_inv = np.exp(log_inv)
    gL = 1.0 / g_inv
    return gamma * (1.0 - gL) / (delta * (L + g_inv))


def check_shore():
    # L = 1 совпадает с трёмя интервалами при β = γ².
    delta = Fraction(1, 10)
    g = shore_gamma(delta, 1)
    beta = (1 - delta) ** 2
    three = beta * delta / (2 - delta)
    if g != three:
        raise RuntimeError("берег L=1 не совпал с формулой трёх интервалов")
    # Строго меньше δE и предел (1−e^{−λ})/λ.
    for delta_f, L in ((0.1, 1), (0.1, 5), (0.05, 8), (0.01, 20), (0.01, 100)):
        ratio = float(shore_gamma(Fraction(delta_f).limit_denominator(1000), L)) / delta_f
        if not (0 < ratio < 1):
            raise RuntimeError("отношение берега вне (0, 1): %s %s %s" % (delta_f, L, ratio))
    # Ориентиры G3: (1−e^{−λ})/λ при λ = 0.2, 0.05, 0.01.
    targets = {0.2: 0.906346, 0.05: 0.975412, 0.01: 0.995017}
    for lam, ref in targets.items():
        val = (1 - np.exp(-lam)) / lam
        if abs(val - ref) > 5e-5:
            raise RuntimeError("предел λ=%s: %s против %s" % (lam, val, ref))
    # Построение при δ=0.1, L=5, D=2: цикл, ENS на норме, поздний час покрыт.
    delta_f = 0.1
    L = 5
    D = 2.0
    gamma = 1 - delta_f
    r = float(shore_gamma(Fraction(1, 10), 5))
    beta = gamma ** (L + 1)
    c = [1.0] + [0.0] * (L + 1)
    d = [0.0, r / (gamma ** L)] + [0.0] * (L - 1) + [beta - r]
    simulate(c, d, 1.0, delta_f, 1.0)
    deficit = [0.0] + [D - r] * L + [beta - r]
    ens = sum(deficit) - sum(d)
    ens_bar = L * D + beta - gamma
    if abs(ens - ens_bar) > 1e-9:
        raise RuntimeError("ENS построения берега %s != %s" % (ens, ens_bar))
    if abs(d[-1] - (beta - r)) > 1e-12 or d[-1] < beta - r - 1e-12:
        raise RuntimeError("поздний час берега не покрыт")
    print("shore ok", "L=1", str(g), "ratio examples",
          round(float(shore_gamma(Fraction(1, 10), 5)) / 0.1, 6))
    return g


def competition_gamma(x, E, Pc, Pd, eta, delta, tol=1e-6):
    """Γ на границе конкуренции: нормы — минимумы ENS и LOLH при r=0."""
    units = [{"E": E, "Pc": Pc, "Pd": Pd, "eta": eta, "delta": delta}]
    ens_star = ref_milp.solve(x, units, 0.0, objective="ens")
    lol_star = ref_milp.solve(x, units, 0.0, objective="lolh")
    if ens_star is None or lol_star is None:
        raise RuntimeError("разделяющий режим при r=0 не решён")
    case = {
        "x": list(x),
        "units": units,
        "ens_limit": float(ens_star),
        "lolh_limit": float(lol_star),
        "tau": 1.0,
        "cyclic": True,
    }
    rE = ref_milp.bisect(case, "E", tol=tol)
    rH = ref_milp.bisect(case, "H", tol=tol)
    rJ = ref_milp.bisect(case, "J", tol=tol)
    gamma_lo = rJ[0] - max(rE[1], rH[1])
    gamma_hi = rJ[1] - max(rE[0], rH[0])
    return {
        "ens": float(ens_star),
        "lolh": float(lol_star),
        "rE": rE,
        "rH": rH,
        "rJ": rJ,
        "lo": gamma_lo,
        "hi": gamma_hi,
    }


def check_three_and_surplus():
    """Формула D2·δ/(2−δ) и режим, где связывает избыток."""
    # Канон черновика: ёмкость и мощность заряда связывают, D2=0.8.
    canon = competition_gamma([1.0, -1.0, -0.8], 1, 1, 1, 1, 0.1)
    target = 0.8 * 0.1 / 1.9  # 4/95
    if not (canon["lo"] - 1e-8 <= target <= canon["hi"] + 1e-8):
        raise RuntimeError("канон 4/95 вне биссекции %s" % canon)
    # Ёмкость связывает, избыток первого часа больше e*.
    cap = competition_gamma([1.0, -1.0, -0.2], 0.5, 1, 1, 1, 0.1)
    target_cap = 0.2 * 0.1 / 1.9
    if not (cap["lo"] - 1e-8 <= target_cap <= cap["hi"] + 1e-8):
        raise RuntimeError("ёмкость вне биссекции %s цель %s" % (cap, target_cap))
    # Избыток связывает: E и P^c велики. η=1 и η=0.5.
    sur = competition_gamma([1.0, -2.0, -0.5], 10, 10, 1, 1, 0.1)
    gamma = 0.9
    target_sur = 0.5 * 0.1 / (1 + gamma + gamma ** 2)
    if not (sur["lo"] - 1e-8 <= target_sur <= sur["hi"] + 1e-8):
        raise RuntimeError("избыток η=1 вне биссекции %s цель %s" % (sur, target_sur))
    sur5 = competition_gamma([1.0, -2.0, -0.2], 10, 10, 1, 0.5, 0.1)
    eta = 0.5
    target5 = 0.2 * 0.1 / (1 + gamma + eta * gamma ** 2)
    if not (sur5["lo"] - 1e-8 <= target5 <= sur5["hi"] + 1e-8):
        raise RuntimeError("избыток η=0.5 вне биссекции %s цель %s" % (sur5, target5))
    print("three canon", canon["rJ"], "target", round(target, 10))
    print("capacity", cap["rJ"], "target", round(target_cap, 10))
    print("surplus η=1", sur["rJ"], "target", round(target_sur, 10))
    print("surplus η=0.5", sur5["rJ"], "target", round(target5, 10))
    return target


def tank_curve():
    deltas = [0.005, 0.01, 0.02, 0.05]
    rows = []
    for delta in deltas:
        print("tank δ=%s" % delta, flush=True)
        row = competition_gamma(DAY, 3, 1, 1, 1, delta, tol=1e-6)
        mid = 0.5 * (row["lo"] + row["hi"])
        print("  ens", row["ens"], "lolh", row["lolh"], "Γ", row["lo"], row["hi"], flush=True)
        if row["hi"] < -1e-9 or row["lo"] > delta * 3 + 1e-6:
            raise RuntimeError("бак вышел за [0, δE]: %s" % row)
        if delta in G2_TANK and abs(row["hi"] - G2_TANK[delta]) > 2e-5:
            raise RuntimeError("бак не совпал с G2: %s против %s" % (row["hi"], G2_TANK[delta]))
        rows.append((delta, mid, row))
    return rows


def draw_quarter(modes):
    setup_style()
    panels = [
        ("H", "Для нормы H, r = 0", modes["H"]),
        ("D", "Для нормы D, r = 0", modes["D"]),
        ("J", "Совместно, r = 1/4", modes["J"]),
    ]
    colors = {"дефицит": "#4C78A8", "разряд": "#F58518", "непокрытая нагрузка": "#E45756"}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), sharey=True)
    hours = np.arange(1, 7)
    width = 0.24
    for ax, (key, title, (deficit, discharge, ell)) in zip(axes, panels):
        defi = np.array([float(v) for v in deficit])
        dis = np.array([float(v) for v in discharge])
        un = np.array([float(v) for v in ell])
        covered = [i for i in range(6) if defi[i] > 1e-12 and un[i] <= 1e-12]
        for i in covered:
            ax.axvspan(hours[i] - 0.48, hours[i] + 0.48, color="#E2F0D9", zorder=0)
        ax.bar(hours - width, defi, width, color=colors["дефицит"], label="дефицит", zorder=2)
        ax.bar(hours, dis, width, color=colors["разряд"], label="разряд", zorder=2)
        ax.bar(hours + width, un, width, color=colors["непокрытая нагрузка"], label="непокрытая нагрузка", zorder=2)
        for a, b in runs_of_positive(un):
            top = max(float(un[i]) for i in range(a, b + 1))
            y = top + 0.10
            x0 = hours[a] - (0.34 if a == b else 0.0)
            x1 = hours[b] + (0.34 if a == b else 0.0)
            ax.plot([x0, x1], [y, y], color="#6B3FA0", lw=2.0, solid_capstyle="butt", zorder=3)
            ax.plot([x0, x0], [y - 0.06, y], color="#6B3FA0", lw=2.0, zorder=3)
            ax.plot([x1, x1], [y - 0.06, y], color="#6B3FA0", lw=2.0, zorder=3)
        ax.text(1, 0.08, "заряд 1", ha="center", va="bottom", fontsize=8, color="#555555")
        ax.set_xlim(0.4, 6.6)
        ax.set_ylim(0, 1.28)
        ax.set_xticks(hours)
        ax.set_xlabel("интервал")
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.6)
        ax.yaxis.grid(True, lw=0.4, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        comma_axis(ax)
    axes[0].set_ylabel("МВт")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles += [
        Patch(facecolor="#E2F0D9", edgecolor="#9BBB8A", label="полностью покрытый час"),
        Line2D([0], [0], color="#6B3FA0", lw=2.0, label="серия непокрытых часов"),
    ]
    labels += ["полностью покрытый час", "серия непокрытых часов"]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = FIG / "fig_quarter.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def draw_shore():
    setup_style()
    deltas = [0.01, 0.05, 0.1]
    L = np.arange(1, 401)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    for delta in deltas:
        ratio = np.array([shore_ratio_float(delta, int(n)) for n in L])
        ax.plot(L, ratio, label="δ = %s" % str(delta).replace(".", ","))
    ax.axhline(1.0, color="black", lw=0.9, ls="--", label="1")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\Gamma/(\delta E)$")
    ax.set_xlim(1, 400)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Как функция длины берега")
    ax.yaxis.grid(True, lw=0.4, alpha=0.35)
    comma_axis(ax)

    ax = axes[1]
    lam = np.linspace(0.02, 4.0, 400)
    ax.plot(lam, (1 - np.exp(-lam)) / lam, color="black", lw=1.6, label=r"$(1-e^{-\lambda})/\lambda$")
    for delta in deltas:
        xs, ys = [], []
        n = 1
        while delta * n <= 4.0 and n <= 2000:
            xs.append(delta * n)
            ys.append(shore_ratio_float(delta, n))
            n += 1
        ax.plot(xs, ys, label="δ = %s" % str(delta).replace(".", ","))
    ax.axhline(1.0, color="black", lw=0.9, ls="--")
    ax.set_xlabel(r"$\lambda=\delta L$")
    ax.set_ylabel(r"$\Gamma/(\delta E)$")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Сжатие к пределу")
    ax.yaxis.grid(True, lw=0.4, alpha=0.35)
    comma_axis(ax)
    fig.tight_layout()
    path = FIG / "fig_shore.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def draw_gamma(tank_rows):
    setup_style()
    delta = np.linspace(0.0, 0.055, 400)
    # Пример черновика: D_2 = 0.8, P^d = 1.
    formula = 0.8 * delta / (2.0 - delta)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(delta, formula, color="#4C78A8", lw=1.8, label=r"три интервала, $D_2=0{,}8$")
    td = [row[0] for row in tank_rows]
    tg = [row[1] for row in tank_rows]
    ax.plot(td, tg, linestyle="--", color="#E45756", lw=1.1, zorder=2)
    ax.plot(
        td, tg, linestyle="none", marker="o", color="#E45756",
        markerfacecolor="#E45756", markeredgecolor="#E45756",
        markersize=8, label="бак, обычные сутки, E = 3", zorder=3,
    )
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\Gamma/P^{d}$")
    ax.set_xlim(0, 0.055)
    ax.set_ylim(0, max(0.025, max(tg) * 1.25))
    ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.yaxis.grid(True, lw=0.4, alpha=0.35)
    ax.legend(frameon=False)
    comma_axis(ax)
    fig.tight_layout()
    path = FIG / "fig_gamma_delta.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    FIG.mkdir(exist_ok=True)
    modes = check_quarter()
    check_shore()
    check_three_and_surplus()
    rows = tank_curve()
    draw_quarter(modes)
    draw_shore()
    draw_gamma(rows)
    print("ALL OK")


if __name__ == "__main__":
    main()
