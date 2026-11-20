"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Reads every <id>.in case in the input directory and writes one <id>.out
forecast per case: five quantiles (0.05/0.25/0.5/0.75/0.95) of each trophy's
Exophase earn rate (%) at the case's target. Python 3.13, standard library
only, no network, deterministic: the same input always yields the same bytes.

Model
-----
Exophase refreshes a game's page on its own schedule, so the history is a
staircase: long runs of identical reads (a frozen page) punctuated by
refreshes. Consecutive identical reads are collapsed into one snapshot; a
snapshot stands for the page as of the last refresh before it.

Per trophy, with at least three distinct snapshots, the median is a
log-time trend: weighted least squares of logit(rate) on
ln(1 + hours since release 2026-11-19T00:00:00Z), over the last 36 hours of
snapshots (the regime that extrapolates to the target), recency-weighted.
The slope is shrunk toward the median slope of the trophy's pace class
(story / strand / grind / other, read off the description), because one
trophy's noisy steps say little on their own, and the move from the last
rate is capped at 3.5 logit units and damped by 0.75: beyond the fitted
window, earn-rate curves saturate, so a log-time line overstates the
rise. Log-time extrapolation already decelerates the rise, which is what
a saturating player base does.

The spread is honest about a multi-day extrapolation: the random-walk term
(0.5 logit per sqrt day), the trend fit's own prediction error at the
target, and a 0.6 floor for everything the trend cannot see, combined in
quadrature, floored at 0.9 and capped at 2.8 logit units.

With fewer than three distinct snapshots the page is stale (the practice
cases: one frozen launch-night page). Then the median uses a dilution
prior: the frozen page is as of roughly the release, the tracked-player
count grows about linearly in (1 + hours since release), and newcomers
earn each trophy at a discounted fraction of its current rate, so rates
fall toward the refresh. A trophy essentially every tracked player has
(>95%) stays near 100%. The spread is wide (1.6) because the next
refresh's size is unknown.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the scored levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Trend window: only the most recent snapshots drive the median.
TREND_WINDOW_HOURS = 36.0
# Recency half-life inside the window, in hours.
RECENCY_HALF_LIFE_HOURS = 24.0
# Prior weight (in pseudo-observations) of the pace-class slope.
SHRINK_WEIGHT = 4.0
# Widest logit move allowed away from the last observed rate.
MAX_LOGIT_MOVE = 3.5
# The log-time line is fitted inside the window; beyond it, earn-rate
# curves saturate, so only this share of the extrapolated move is kept.
EXTRAPOLATION_DAMP = 0.75
# Random-walk spread, logit units per sqrt day.
RW_SIGMA = 0.5
# Extra spread for model error the trend cannot see.
MODEL_SIGMA = 0.6
MIN_SIGMA = 0.9
MAX_SIGMA = 2.8
# Flat-history branch: spread and newcomer earn discount.
FLAT_SIGMA = 1.6
NEWCOMER_DISCOUNT = 0.25
UNIVERSAL_RATE = 95.0


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = math.floor(x * 10000 + 0.5) / 10000
    return int(scaled) if float(scaled).is_integer() else scaled


def hours_since_release(moment):
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def pace_class(trophy):
    """How fast this trophy's earn rate plausibly moves, from its text."""
    text = (trophy["name"] + " " + trophy["description"]).lower()
    if "chapter" in text or "final mission" in text:
        return "story"
    if "mission for" in text or "mission strand" in text:
        return "strand"
    if "100% completion" in text:
        return "grind"
    match = re.search(r"\b(\d{2,3})\b", text)
    if match and int(match.group(1)) >= 30:
        return "grind"
    return "other"


def distinct_snapshots(case):
    """Collapse runs of identical reads into one snapshot each.

    Returns a list of dicts {at, x, players, rates}, oldest first, where x
    is ln(1 + hours since release) at the read and rates follow the case's
    trophy order. The snapshot is dated at its first occurrence.
    """
    snapshots = []
    previous_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == previous_key:
            continue
        previous_key = key
        moment = parse_time(read["at"])
        snapshots.append({
            "at": moment,
            "x": math.log1p(hours_since_release(moment)),
            "players": read["players_tracked"],
            "rates": list(read["rates"]),
        })
    return snapshots


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, s2, n, xbar, sxx).

    s2 is the weighted residual variance (0 when the fit is exact or the
    points are degenerate); xbar and sxx are the weighted mean and spread
    of x, for the prediction error at a new x.
    """
    n = len(xs)
    wsum = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / wsum
    ybar = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return ybar, 0.0, 0.0, n, xbar, sxx
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    dof = max(n - 2, 1)
    s2 = sse / (wsum * dof / n) if wsum > 0 else 0.0
    return a, b, max(s2, 0.0), n, xbar, sxx


def median_of(values):
    ordered = sorted(values)
    m = len(ordered)
    if m == 0:
        return 0.0
    if m % 2:
        return ordered[m // 2]
    return 0.5 * (ordered[m // 2 - 1] + ordered[m // 2])


def trend_forecast(trophies, snapshots, x_target, horizon_days):
    """Median logit + spread per trophy from the log-time trend.

    Returns a list of (median_logit, sigma) aligned with `trophies`.
    """
    last = snapshots[-1]
    last_moment = last["at"]
    window_start = last_moment.timestamp() - TREND_WINDOW_HOURS * 3600
    window = [s for s in snapshots if s["at"].timestamp() >= window_start]
    if len(window) < 3:
        window = snapshots

    # Fit one weighted line per trophy over the window.
    fits = []
    for i in range(len(trophies)):
        xs, ys, ws = [], [], []
        for snap in window:
            rate = snap["rates"][i]
            if rate is None:
                continue
            xs.append(snap["x"])
            ys.append(logit(rate))
            age_h = (last_moment - snap["at"]).total_seconds() / 3600.0
            ws.append(0.5 ** (age_h / RECENCY_HALF_LIFE_HOURS))
        if len(xs) >= 3:
            fits.append(weighted_line(xs, ys, ws))
        else:
            fits.append(None)

    # Median slope per pace class, as the shrinkage target.
    slopes_by_class = {}
    for trophy, fit in zip(trophies, fits):
        if fit is None:
            continue
        slopes_by_class.setdefault(pace_class(trophy), []).append(fit[1])
    class_slope = {c: median_of(v) for c, v in slopes_by_class.items()}
    global_slope = median_of([f[1] for f in fits if f is not None])

    out = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        if last_rate is None:
            for snap in reversed(snapshots):
                if snap["rates"][i] is not None:
                    last_rate = snap["rates"][i]
                    break
        if last_rate is None:
            last_rate = 0.0
        last_logit = logit(last_rate)
        fit = fits[i]
        if fit is None:
            out.append((last_logit, FLAT_SIGMA))
            continue
        a, b, s2, n, xbar, sxx = fit
        prior = class_slope.get(pace_class(trophy), global_slope)
        b_used = (n * b + SHRINK_WEIGHT * prior) / (n + SHRINK_WEIGHT)
        # The line passes through (xbar, a + b*xbar); the shrunk slope
        # pivots around that point, not the origin.
        y_hat = (a + b * xbar) + b_used * (x_target - xbar)
        move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, y_hat - last_logit))
        median_logit = last_logit + EXTRAPOLATION_DAMP * move
        # Prediction error at the target: residual scatter, leverage, plus
        # the random walk and the model-error floor, in quadrature.
        leverage = 1.0 + 1.0 / n + (x_target - xbar) ** 2 / sxx if sxx > 1e-12 else 1.0
        se_pred = math.sqrt(max(s2, 0.0025) * leverage)
        rw = RW_SIGMA * math.sqrt(horizon_days)
        sigma = math.sqrt(rw * rw + se_pred * se_pred + MODEL_SIGMA ** 2)
        sigma = max(MIN_SIGMA, min(MAX_SIGMA, sigma))
        out.append((median_logit, sigma))
    return out


def flat_forecast(trophies, snapshots, x_target, horizon_days, target_moment):
    """Median logit + spread when the page has never moved.

    The frozen page is as of roughly the release, so the tracked-player
    count at the target is the frozen count scaled by (1 + hours since
    release); newcomers earn each trophy at a discounted fraction of its
    current rate, which dilutes every rate except a near-universal one.
    """
    snap = snapshots[-1]
    first_moment = snapshots[0]["at"]
    h_first = hours_since_release(first_moment)
    h_target = hours_since_release(target_moment)
    growth = (1.0 + h_target) / max(1.0 + h_first, 1.0)
    players_target = snap["players"] * growth
    rows = []
    for i in range(len(trophies)):
        rate = snap["rates"][i]
        if rate is None:
            rate = 0.0
        if rate >= UNIVERSAL_RATE:
            median_logit = logit(99.9)
        else:
            earners = rate / 100.0 * snap["players"]
            newcomers = max(players_target - snap["players"], 0.0)
            earners += NEWCOMER_DISCOUNT * (rate / 100.0) * newcomers
            median_rate = 100.0 * earners / max(players_target, 1.0)
            median_logit = logit(median_rate)
        rows.append((median_logit, FLAT_SIGMA))
    return rows


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    snapshots = distinct_snapshots(case)
    target_moment = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target_moment))
    horizon_days = case["horizon_minutes"] / 1440.0

    if len(snapshots) >= 3:
        forecasts = trend_forecast(trophies, snapshots, x_target, horizon_days)
    else:
        forecasts = flat_forecast(trophies, snapshots, x_target,
                                  horizon_days, target_moment)

    rows = []
    for trophy, (median_logit, sigma) in zip(trophies, forecasts):
        qs = []
        previous = None
        for z in Z:
            q = round4(sigmoid(median_logit + z * sigma))
            q = min(100.0, max(0.0, q))
            if previous is not None and q < previous:
                q = previous
            qs.append(q)
            previous = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
