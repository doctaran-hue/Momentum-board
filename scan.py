"""
Daily scan job for the NSE momentum board.

Runs headless (GitHub Actions), writes docs/data/latest.json which the static
dashboard reads. Keeps the previous run so it can report what changed today.

  python scan.py            # live, needs network
  python scan.py --demo     # synthetic data, no network, for previewing the UI

Outputs
  docs/data/latest.json               current board
  docs/data/history/YYYY-MM-DD.json   archive, one per run
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd


# ====================================================================
# ENGINE
# ====================================================================
# ----------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------
def add_indicators(df):
    c = df["Close"]
    df["SMA50"] = c.rolling(50).mean()
    df["SMA150"] = c.rolling(150).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["SMA200_slope"] = df["SMA200"] - df["SMA200"].shift(21)
    df["HI252"] = df["High"].rolling(252).max()
    df["LO252"] = df["Low"].rolling(252).min()
    df["VOL50"] = df["Volume"].rolling(50).mean()
    return df


def stage2_gate(df, i):
    """Minervini-style trend template evaluated at row i. Returns (bool, dict)."""
    r = df.iloc[i]
    if pd.isna(r["SMA200"]) or pd.isna(r["LO252"]) or r["LO252"] <= 0:
        return False, {}
    c = r["Close"]
    checks = {
        "px>SMA150": c > r["SMA150"],
        "px>SMA200": c > r["SMA200"],
        "SMA150>SMA200": r["SMA150"] > r["SMA200"],
        "SMA200_rising": r["SMA200_slope"] > 0,
        "SMA50>SMA150": r["SMA50"] > r["SMA150"],
        "px>SMA50": c > r["SMA50"],
        "px>=30pct_off_low": (c / r["LO252"] - 1) >= 0.30,
        "px_within_25pct_hi": (c / r["HI252"] - 1) >= -0.25,
    }
    return all(checks.values()), checks


# ----------------------------------------------------------------------
# Base / advance segmentation
# ----------------------------------------------------------------------
def segment(df, min_len=25, min_depth=0.08, max_depth=0.35):
    """
    Walk the series. Identify peak -> consolidation -> breakout segments.

    A segment qualifies as a BASE if it lasted >= min_len sessions without
    exceeding the peak, and its drawdown from the peak sits in
    [min_depth, max_depth]. Shorter or shallower pauses are treated as noise
    within an ongoing advance.

    Returns (bases, open_base) where open_base describes an unresolved
    consolidation running to the end of the data (or None).
    """
    high = df["High"].values
    low = df["Low"].values
    n = len(high)
    if n < 2:
        return [], None

    bases = []
    peak_idx, peak_val = 0, high[0]

    while True:
        # Walk up while new highs are being made.
        j = peak_idx + 1
        while j < n and high[j] >= peak_val:
            peak_val = high[j]
            peak_idx = j
            j += 1
        if j >= n:
            return bases, None  # ended at a new high, no open base

        # j is the first bar that failed to exceed the peak. Find breakout.
        k = j
        while k < n and high[k] < peak_val:
            k += 1

        resolved = k < n
        seg_end = (k - 1) if resolved else (n - 1)
        length = seg_end - peak_idx
        trough = low[peak_idx : seg_end + 1].min()
        depth = (peak_val - trough) / peak_val

        qualifies = (length >= min_len) and (min_depth <= depth <= max_depth)

        if not resolved:
            open_base = {
                "peak_idx": peak_idx,
                "peak_val": peak_val,
                "start_date": df.index[peak_idx],
                "length": length,
                "depth": depth,
                "trough": trough,
                "qualifies_len": length >= min_len,
                "qualifies_depth": min_depth <= depth <= max_depth,
            }
            return bases, open_base

        if qualifies:
            bases.append(
                {
                    "peak_idx": peak_idx,
                    "peak_val": peak_val,          # this is the pivot
                    "start_date": df.index[peak_idx],
                    "breakout_idx": k,
                    "breakout_date": df.index[k],
                    "length": length,
                    "depth": depth,
                    "trough": trough,
                }
            )

        # Continue from the breakout bar.
        peak_idx, peak_val = k, high[k]
        if peak_idx >= n - 1:
            return bases, None


def attach_advances(bases, open_base, df):
    """
    Advance after each base is measured from its pivot to the next pivot
    (or to the running high if it's the last completed base).
    """
    high = df["High"].values
    for m, b in enumerate(bases):
        if m + 1 < len(bases):
            nxt = bases[m + 1]["peak_val"]
        elif open_base is not None:
            nxt = open_base["peak_val"]
        else:
            nxt = high[b["breakout_idx"] :].max()
        b["advance"] = nxt / b["peak_val"] - 1
    return bases


# ----------------------------------------------------------------------
# Reset detection (Stage 4 / Stage 1 re-entry)
# ----------------------------------------------------------------------
def last_reset_idx(df, below_days=126, dd_thresh=0.50):
    """
    Reset fires on EITHER:
      (a) >= below_days sessions with close < SMA200 AND SMA200 declining
      (b) drawdown >= dd_thresh from the running peak

    Returns the index at which the most recent reset COMPLETED, or -1.
    """
    c = df["Close"].values
    s200 = df["SMA200"].values
    slope = df["SMA200_slope"].values
    n = len(c)

    bad = np.zeros(n, dtype=bool)
    valid = ~(np.isnan(s200) | np.isnan(slope))
    bad[valid] = (c[valid] < s200[valid]) & (slope[valid] < 0)

    reset = -1
    run = 0
    for i in range(n):
        run = run + 1 if bad[i] else 0
        if run >= below_days:
            reset = i

    runmax = np.maximum.accumulate(np.where(np.isnan(c), -np.inf, c))
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = 1 - c / runmax
    dd_hits = np.where(dd >= dd_thresh)[0]
    if len(dd_hits):
        reset = max(reset, int(dd_hits[-1]))

    return reset


# ----------------------------------------------------------------------
# Tightness / contraction / volume metrics on the open base
# ----------------------------------------------------------------------
def count_start_idx(df, reset):
    """
    Bases are counted from the Stage 1 low, i.e. the lowest close AFTER the
    reset completed. Segmenting from before the reset is meaningless: the
    pre-crash high is never exceeded, so the whole recovery reads as one
    enormous unresolved base.
    """
    n = len(df)
    if reset < 0:
        return 0, False  # no reset in window -> count is unreliable
    c = df["Close"].values

    # The reset *completes* well after the actual low (the 126-day count runs
    # on into the early recovery). So search back to the peak that preceded
    # the decline, and forward a little in case a drawdown-triggered reset
    # fired before the true bottom.
    pre_peak = int(np.nanargmax(c[: reset + 1]))
    hi_end = min(reset + 60, n - 1)
    if hi_end <= pre_peak:
        return min(reset, n - 1), False
    lo = int(np.nanargmin(c[pre_peak : hi_end + 1])) + pre_peak
    return lo, True


def analyse_stock(df, min_len=25, min_depth=0.08, max_depth=0.35,
                  min_advance=0.20, below_days=126, dd_thresh=0.50):
    """Full pipeline for one symbol. Returns a dict."""
    reset = last_reset_idx(df, below_days, dd_thresh)
    start, reliable = count_start_idx(df, reset)
    sub = df.iloc[start:]
    bases, open_base = segment(sub, min_len, min_depth, max_depth)
    bases = attach_advances(bases, open_base, sub)
    qual = [b for b in bases if b["advance"] >= min_advance]
    return {
        "reset_idx": reset,
        "reset_date": df.index[reset] if reset >= 0 else None,
        "count_start": df.index[start],
        "count_reliable": reliable and (start > 0),
        "bases": bases,
        "qual_bases": qual,
        "open_base": open_base,
        "next_base_number": len(qual) + 1,
        "sub": sub,
    }


def pullback_depths(seg, thresh=0.03):
    """
    Successive pullback depths inside a base, via a percentage zigzag.

    Equal-time thirds do NOT measure this: a V-shaped base is wide-narrow-wide
    by construction, so thirds can never be monotonic even when the base is
    textbook. VCP contraction is about successive *retracement legs*, so the
    swings have to be found first.

    Returns (depths, n_swings) where depths are high->low retracements in order.
    """
    hi = seg["High"].values
    lo = seg["Low"].values
    n = len(hi)
    if n < 6:
        return [], 0

    # zigzag: alternate between confirmed swing highs and swing lows
    piv = []                      # (index, price, kind)
    direction = 0                 # +1 seeking high, -1 seeking low
    ext_i, ext_p = 0, hi[0]
    for i in range(1, n):
        if direction >= 0:
            if hi[i] > ext_p:
                ext_i, ext_p = i, hi[i]
            elif lo[i] <= ext_p * (1 - thresh):
                piv.append((ext_i, ext_p, "H"))
                direction = -1
                ext_i, ext_p = i, lo[i]
        else:
            if lo[i] < ext_p:
                ext_i, ext_p = i, lo[i]
            elif hi[i] >= ext_p * (1 + thresh):
                piv.append((ext_i, ext_p, "L"))
                direction = 1
                ext_i, ext_p = i, hi[i]
    piv.append((ext_i, ext_p, "H" if direction >= 0 else "L"))

    depths = []
    for a, b in zip(piv, piv[1:]):
        if a[2] == "H" and b[2] == "L" and a[1] > 0:
            depths.append((a[1] - b[1]) / a[1])
    return depths, len(piv)


def base_metrics(df, open_base):
    a = open_base["peak_idx"]
    b = len(df) - 1
    seg = df.iloc[a : b + 1]
    px = float(seg["Close"].iloc[-1])

    # Cheap fields are always computable, even for a 2-bar base. Returning an
    # empty dict here (as this used to) makes every caller crash on a stock that
    # topped out three sessions ago.
    out = {
        "assessable": len(seg) >= 6,
        "n_bars": int(len(seg)),
        "range_contraction": np.nan,
        "pullbacks": [],
        "n_pullbacks": 0,
        "n_swings": 0,
        "contracting": None,
        "net_contract": None,
        "vol_dryup": np.nan,
        "pct_below_pivot": float(px / open_base["peak_val"] - 1),
        "above_SMA50": bool(px > df["SMA50"].iloc[-1]),
        "pos_in_base": (
            float((px - open_base["trough"])
                  / (open_base["peak_val"] - open_base["trough"]))
            if open_base["peak_val"] > open_base["trough"] else np.nan
        ),
    }
    if not out["assessable"]:
        return out

    rng = (seg["High"] - seg["Low"]) / seg["Close"]
    third = max(2, len(seg) // 3)
    r1 = rng.iloc[:third].mean()
    r3 = rng.iloc[-third:].mean()

    # successive pullback depths (proper VCP construction, see pullback_depths)
    pbs, n_swings = pullback_depths(seg)
    vol_base = seg["Volume"].mean()
    vol_recent = seg["Volume"].iloc[-5:].mean()

    out.update({
        "range_contraction": float(r3 / r1) if r1 > 0 else np.nan,
        "pullbacks": [round(x * 100, 1) for x in pbs],
        "n_pullbacks": len(pbs),
        "n_swings": n_swings,
        "contracting": (
            all(pbs[i] > pbs[i + 1] for i in range(len(pbs) - 1))
            if len(pbs) >= 2 else None          # None = not evaluable
        ),
        "net_contract": (pbs[-1] < pbs[0]) if len(pbs) >= 2 else None,
        "vol_dryup": float(vol_recent / vol_base) if vol_base > 0 else np.nan,
    })
    return out


# ====================================================================
# QUALITY GATES + CONFOUND TEST
# ====================================================================
# Which flags are HARD gates. Anything here must pass for tier == "PASS".
HARD_GATES = ["q_ranges", "q_volume", "q_contract"]

# All flags, in display order.
ALL_FLAGS = ["q_ranges", "q_volume", "q_contract", "q_net_contract",
             "q_depth", "q_upper", "q_sma50"]

FLAG_DESC = {
    "q_ranges":       "daily ranges not expanding (range_contraction < range_max)",
    "q_volume":       "volume not expanding (vol_dryup < vol_max)",
    "q_contract":     "successive pullbacks monotonically shrinking (vacuous if <2)",
    "q_net_contract": "final pullback shallower than the first",
    "q_depth":        "base depth within the tight band",
    "q_upper":        "price in the upper half of the base",
    "q_sma50":        "price above the 50-DMA",
}


def quality_flags(m, depth, max_quality_depth=0.20,
                  range_max=1.15, vol_max=1.20):
    """
    m: dict from base_metrics(). depth: open_base['depth'].
    Returns dict of boolean flags + score + tier.
    """
    con = m.get("pullbacks") or []
    rc = m.get("range_contraction", np.nan)
    vd = m.get("vol_dryup", np.nan)
    pos = m.get("pos_in_base", np.nan)

    # Thresholds sit above 1.0 deliberately. The gate exists to reject bases
    # whose ranges/volume are meaningfully EXPANDING, not to demand strict
    # contraction - a base with flat daily ranges is fine, one with ranges 60%
    # wider in its final third is not. A knife-edge at exactly 1.0 rejects
    # textbook bases on noise.
    f = {
        "q_ranges": bool(np.isfinite(rc) and rc < range_max),
        "q_volume": bool(np.isfinite(vd) and vd < vol_max),
        # With fewer than 2 pullbacks there is nothing to compare, so these
        # pass rather than fail - a tight base with a single retracement is not
        # a defect. n_pullbacks is surfaced so the vacuous case is visible.
        "q_contract": (True if len(con) < 2 else
                       all(con[i] > con[i + 1] for i in range(len(con) - 1))),
        "q_net_contract": (True if len(con) < 2 else con[-1] < con[0]),
        "q_depth": bool(np.isfinite(depth) and depth <= max_quality_depth),
        "q_upper": bool(np.isfinite(pos) and pos >= 0.5),
        "q_sma50": bool(m.get("above_SMA50", False)),
    }
    f["quality_score"] = int(sum(f[k] for k in ALL_FLAGS))
    missing_hard = [k for k in HARD_GATES if not f[k]]
    f["hard_fails"] = ",".join(k.replace("q_", "") for k in missing_hard)
    f["tier"] = ("PASS" if not missing_hard
                 else "NEAR" if len(missing_hard) == 1
                 else "FAIL")
    return f


# ----------------------------------------------------------------------
# Confound test
# ----------------------------------------------------------------------
def index_proxy(panel, min_names=20):
    """
    Equal-weight index proxy built from the panel itself: median cross-sectional
    daily return, cumulated. Median rather than mean so a few outliers cannot
    drag it. Used to locate market-wide drawdown troughs without needing an
    external index series.
    """
    rets = {}
    for s, d in panel.items():
        rets[s] = d["Close"].pct_change()
    R = pd.DataFrame(rets)
    n = R.notna().sum(axis=1)
    med = R.median(axis=1).where(n >= min_names)
    med = med.dropna()
    if len(med) < 200:
        return None
    return (1 + med).cumprod()


def drawdown_troughs(proxy, min_dd=0.10):
    """
    Locate the trough of every episode where the proxy fell >= min_dd from its
    running peak. Returns a DatetimeIndex of trough dates.
    """
    if proxy is None or len(proxy) < 200:
        return pd.DatetimeIndex([])
    runmax = proxy.cummax()
    dd = 1 - proxy / runmax
    in_ep = dd >= min_dd
    troughs = []
    i, n = 0, len(proxy)
    arr = proxy.values
    while i < n:
        if not in_ep.iloc[i]:
            i += 1
            continue
        j = i
        while j < n and in_ep.iloc[j]:
            j += 1
        # extend to the local minimum: episode start is where dd began rising
        k0 = i
        while k0 > 0 and dd.iloc[k0 - 1] > 0:
            k0 -= 1
        seg = arr[k0:j]
        troughs.append(proxy.index[k0 + int(np.argmin(seg))])
        i = j
    return pd.DatetimeIndex(sorted(set(troughs)))


def confound_report(scan, panel, window=60, verbose=True):
    """
    Tests whether count_start is really a stock-specific variable or just
    time-since-market-bottom.

    Reports:
      top2_year_share  - a bimodal distribution can sit at 31% max and still be
                         two spikes; the max-year statistic misses that.
      monthly_hhi      - Herfindahl on monthly shares. 1/n_months = uniform.
      trough_share     - THE test: fraction of count_starts landing within
                         +/- `window` sessions of a proxy drawdown trough.
    """
    sub = scan[scan.get("in_base", pd.Series(dtype=bool)).fillna(False)
               & scan.get("reliable", scan.get("count_reliable", True))]
    if not len(sub):
        return None
    cs = pd.to_datetime(sub["count_start"])

    by_year = cs.dt.year.value_counts().sort_index()
    yr_sh = (by_year / by_year.sum()).sort_values(ascending=False)
    top1 = float(yr_sh.iloc[0])
    top2 = float(yr_sh.iloc[:2].sum()) if len(yr_sh) > 1 else top1

    by_month = cs.dt.to_period("M").value_counts()
    hhi = float(((by_month / by_month.sum()) ** 2).sum())
    n_months = int(by_month.shape[0])

    proxy = index_proxy(panel)
    troughs = drawdown_troughs(proxy)
    trough_share = np.nan
    dists = None
    if proxy is not None and len(troughs):
        pos = pd.Series(np.arange(len(proxy)), index=proxy.index)
        tpos = np.array([pos.get(t) for t in troughs if t in pos.index])
        cpos = []
        for d in cs:
            idx = pos.index.searchsorted(d)
            cpos.append(min(max(idx, 0), len(pos) - 1))
        cpos = np.array(cpos)
        dists = np.min(np.abs(cpos[:, None] - tpos[None, :]), axis=1)
        trough_share = float((dists <= window).mean())
    elif proxy is not None:
        # Proxy built fine, but there were no market-wide drawdown episodes.
        # That is not a failure - it means the confound cannot arise here.
        trough_share = 0.0

    out = dict(by_year=by_year, top1_year_share=top1, top2_year_share=top2,
               monthly_hhi=hhi, n_months=n_months, uniform_hhi=1.0 / max(n_months, 1),
               troughs=troughs, trough_share=trough_share, distances=dists,
               proxy_built=proxy is not None)

    if verbose:
        print("--- count_start by year ---")
        print(by_year.to_string())
        print(f"\ntop-1 year share      : {top1:.0%}")
        print(f"top-2 year share      : {top2:.0%}   "
              "<- a bimodal distribution hides behind a low top-1")
        print(f"monthly HHI           : {hhi:.3f}  (uniform would be "
              f"{1.0/max(n_months,1):.3f} across {n_months} months)")
        if proxy is None:
            print("index proxy could not be built (too few overlapping names); "
                  "trough test skipped")
        elif not len(troughs):
            print("proxy drawdown troughs: none >= 10% in this window")
            print("--> confound cannot arise: no market-wide declines to cluster on")
        else:
            print(f"proxy drawdown troughs: "
                  f"{', '.join(str(t.date()) for t in troughs)}")
            print(f"within +/-{window} sessions of a trough: {trough_share:.0%}")
            verdict = ("CONFOUNDED - count_start is tracking market bottoms"
                       if trough_share > 0.60 else
                       "PARTIALLY CONFOUNDED - control for reset cohort"
                       if trough_share > 0.35 else
                       "largely idiosyncratic")
            print(f"--> {verdict}")
        print("\nThe fix is not to discard the count - it is to compare base")
        print("numbers WITHIN a reset cohort, where time-since-bottom is held")
        print("constant by construction. Cohort table below.")
    return out


def cohort_table(scan):
    """
    Reset cohort x base number. Comparing base 2 against base 4 down a single
    column holds time-since-bottom constant.
    """
    sub = scan[scan.get("in_base", pd.Series(dtype=bool)).fillna(False)]
    if not len(sub):
        return None
    t = sub.copy()
    t["cohort"] = pd.to_datetime(t["count_start"]).dt.year
    return pd.crosstab(t["cohort"], t["next_base"])


# ====================================================================
# SCAN
# ====================================================================
IST = timezone(timedelta(hours=5, minutes=30))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "docs", "data")
LATEST = os.path.join(DATA, "latest.json")

# ---- knobs -------------------------------------------------------------
YEARS = 12
BENCH = "^NSEI"              # Nifty 50, for relative-strength and regime
RS_LONG, RS_SHORT = 123, 55
RS_TREND_LOOKBACK = 21       # sessions back, for RS rising/falling
RS_TREND_BAND = 5.0          # percentile points that count as a real move

# ---- Anchored RS (ARS), ported from the RS Screener v2 notebook ----
HIGH_LOOKBACK    = 252       # 52-week high window
PROX_MAIN        = 0.85      # main lists: within 15% of the 52w high
PROX_EMERGING    = 0.75      # EMERGING relaxed
FAKE_CONC_LIMIT  = 0.50      # top-3 up days >50% of the up-move = fake
TOP_N            = 25        # rolling RS list size
RECOVERY_BARS    = 15        # bars above 200 EMA to confirm recovery (whipsaw fix)
MANUAL_ANCHOR    = None      # "2026-02-27" to force; None = auto-detect
ARS_STREAK_MIN   = 21        # consecutive ARS-positive days for EMERGING
ARS_IMPROVE_LAG  = 10        # ARS today must beat ARS this many days ago
YOUNG_STREAK_MAX = 60        # cap streak to exclude sideways defensives
USE_YOUNG_FILTER = True
DD_DROP = 0.002              # index down >0.2% = candidate distribution day
DD_WINDOW = 25               # trailing sessions for the distribution count
RS_LEADER_PCTL = 90          # top decile counts as a leader
BASE_CFG = dict(min_len=25, min_depth=0.08, max_depth=0.35,
                min_advance=0.20, below_days=126, dd_thresh=0.50)
MIN_FORMING, MAX_PCT_BELOW = 15, 0.15
MIN_TURNOVER_CR, BASE_RS_MIN = 5.0, 70
QUAL = dict(max_quality_depth=0.20, range_max=1.15, vol_max=1.20)

FALLBACK = """RELIANCE TCS HDFCBANK ICICIBANK INFY HINDUNILVR ITC SBIN BHARTIARTL BAJFINANCE
KOTAKBANK LT LICI HCLTECH ASIANPAINT AXISBANK MARUTI SUNPHARMA TITAN DMART
ULTRACEMCO ADANIENT WIPRO ONGC NTPC JSWSTEEL POWERGRID M&M TATAMOTORS COALINDIA
BAJAJFINSV ADANIPORTS TATASTEEL HINDALCO SIEMENS PIDILITIND GRASIM TECHM DIVISLAB BRITANNIA
NESTLEIND CIPLA DRREDDY EICHERMOT BPCL SBILIFE HDFCLIFE BAJAJ-AUTO INDUSINDBK APOLLOHOSP
TATACONSUM HEROMOTOCO SHREECEM UPL VEDL GAIL IOC HAVELLS DABUR GODREJCP
AMBUJACEM DLF BANKBARODA PNB CANBK IDFCFIRSTB FEDERALBNK AUBANK BANDHANBNK CHOLAFIN
MUTHOOTFIN LTF SHRIRAMFIN ICICIGI ICICIPRULI MFSL TORNTPHARM ALKEM LUPIN AUROPHARMA
BIOCON ZYDUSLIFE GLENMARK IPCALAB LAURUSLABS SYNGENE ABBOTINDIA PFIZER GLAXO SANOFI
TVSMOTOR ASHOKLEY BALKRISIND MRF APOLLOTYRE BHARATFORG SONACOMS MOTHERSON BOSCHLTD EXIDEIND
TRENT ABFRL PAGEIND VBL COLPAL MARICO EMAMILTD JUBLFOOD DEVYANI WESTLIFE
PERSISTENT COFORGE MPHASIS LTIM KPITTECH TATAELXSI CYIENT OFSS ZENSARTECH SONATSOFTW
POLYCAB KEI CGPOWER ABB THERMAX BHEL CUMMINSIND AIAENG GRINDWELL SCHAEFFLER
JINDALSTEL SAIL NMDC NATIONALUM JSL APLAPOLLO RATNAMANI WELCORP TATACHEM DEEPAKNTR
PIIND SRF NAVINFLUOR AARTIIND ATUL VINATIORGA FLUOROCHEM CLEAN GALAXYSURF SUDARSCHEM
INDIGO IRCTC CONCOR GESHIP ADANIPOWER TATAPOWER TORNTPOWER JSWENERGY NHPC SJVN
LODHA OBEROIRLTY GODREJPROP PRESTIGE PHOENIXLTD BRIGADE SOBHA SUNTECK ANANTRAJ
BEL HAL BDL MAZDOCK COCHINSHIP GRSE DATAPATTNS ZENTEC
DIXON AMBER KAYNES SYRMA CDSL BSE MCX ANGELONE IEX KFINTECH
INDHOTEL CHALET LEMONTREE EIHOTEL RADICO UBL MCDOWELL-N SULA
KAJARIACER CERA SUPREMEIND ASTRAL FINPIPE PRINCEPIPE TIMKEN ASTRAZEN
MANKIND JBCHEPHARM ERIS AJANTPHARM NATCOPHARM GRANULES CAPLIPOINT
POLYMED KRBL AVANTIFEED GODREJAGRO CHAMBLFERT COROMANDEL RALLIS
BLUESTARCO VOLTAS WHIRLPOOL CROMPTON VGUARD
TIINDIA CARBORUNIV ELGIEQUIP KIRLOSENG KIRLOSBROS
JYOTHYLAB HONASA GODFRYPHLP VSTIND
INDIAMART NAUKRI NYKAA ZOMATO PAYTM POLICYBZR
MAXHEALTH FORTIS NH KIMS ASTERDM METROPOLIS LALPATHLAB
GUJGASLTD MGL IGL PETRONET GSPL AEGISCHEM
JSWINFRA GPIL LLOYDSME SHYAMMETL"""


# ----------------------------------------------------------------------
# Sector map for the built-in universe. A universe.csv with an Industry column
# overrides this entirely, which is the better route for a large universe.
SECTORS = {}
_SEC_RAW = {
 "Financials": """HDFCBANK ICICIBANK SBIN KOTAKBANK AXISBANK BAJFINANCE BAJAJFINSV INDUSINDBK
   BANKBARODA PNB CANBK IDFCFIRSTB FEDERALBNK AUBANK BANDHANBNK CHOLAFIN MUTHOOTFIN LTF
   SHRIRAMFIN ICICIGI ICICIPRULI MFSL SBILIFE HDFCLIFE LICI CDSL BSE MCX ANGELONE IEX
   KFINTECH POLICYBZR PAYTM""",
 "Information Technology": """TCS INFY HCLTECH WIPRO TECHM LTIM PERSISTENT COFORGE MPHASIS
   KPITTECH TATAELXSI CYIENT OFSS ZENSARTECH SONATSOFTW""",
 "Oil Gas & Consumable Fuels": """RELIANCE ONGC BPCL IOC GAIL PETRONET GUJGASLTD MGL IGL
   GSPL AEGISCHEM COALINDIA""",
 "Fast Moving Consumer Goods": """HINDUNILVR ITC NESTLEIND BRITANNIA DABUR GODREJCP MARICO
   COLPAL EMAMILTD TATACONSUM VBL RADICO UBL MCDOWELL-N SULA JYOTHYLAB HONASA GODFRYPHLP
   VSTIND KRBL AVANTIFEED GODREJAGRO""",
 "Healthcare": """SUNPHARMA CIPLA DRREDDY DIVISLAB TORNTPHARM ALKEM LUPIN AUROPHARMA BIOCON
   ZYDUSLIFE GLENMARK IPCALAB LAURUSLABS SYNGENE ABBOTINDIA PFIZER GLAXO SANOFI ASTRAZEN
   MANKIND JBCHEPHARM ERIS AJANTPHARM NATCOPHARM GRANULES CAPLIPOINT POLYMED APOLLOHOSP
   MAXHEALTH FORTIS NH KIMS ASTERDM METROPOLIS LALPATHLAB""",
 "Automobile & Components": """MARUTI TATAMOTORS M&M BAJAJ-AUTO HEROMOTOCO EICHERMOT TVSMOTOR
   ASHOKLEY BALKRISIND MRF APOLLOTYRE BHARATFORG SONACOMS MOTHERSON BOSCHLTD EXIDEIND
   TIINDIA""",
 "Capital Goods": """LT SIEMENS ABB THERMAX BHEL CUMMINSIND AIAENG GRINDWELL SCHAEFFLER
   POLYCAB KEI CGPOWER TIMKEN CARBORUNIV ELGIEQUIP KIRLOSENG KIRLOSBROS BLUESTARCO VOLTAS
   WHIRLPOOL CROMPTON VGUARD DIXON AMBER KAYNES SYRMA HAVELLS""",
 "Metals & Mining": """TATASTEEL JSWSTEEL HINDALCO VEDL JINDALSTEL SAIL NMDC NATIONALUM JSL
   APLAPOLLO RATNAMANI WELCORP GPIL LLOYDSME SHYAMMETL""",
 "Chemicals": """PIDILITIND UPL TATACHEM DEEPAKNTR PIIND SRF NAVINFLUOR AARTIIND ATUL
   VINATIORGA FLUOROCHEM CLEAN GALAXYSURF SUDARSCHEM CHAMBLFERT COROMANDEL RALLIS""",
 "Construction Materials": """ULTRACEMCO GRASIM SHREECEM AMBUJACEM KAJARIACER CERA
   SUPREMEIND ASTRAL FINPIPE PRINCEPIPE""",
 "Realty": """DLF LODHA OBEROIRLTY GODREJPROP PRESTIGE PHOENIXLTD BRIGADE SOBHA SUNTECK
   ANANTRAJ""",
 "Power": """NTPC POWERGRID ADANIPOWER TATAPOWER TORNTPOWER JSWENERGY NHPC SJVN""",
 "Services": """ADANIENT ADANIPORTS INDIGO IRCTC CONCOR GESHIP JSWINFRA DMART TRENT ABFRL
   PAGEIND JUBLFOOD DEVYANI WESTLIFE INDHOTEL CHALET LEMONTREE EIHOTEL INDIAMART NAUKRI
   NYKAA ZOMATO""",
 "Telecommunication": """BHARTIARTL""",
 "Consumer Durables": """TITAN ASIANPAINT""",
 "Capital Goods - Defence": """BEL HAL BDL MAZDOCK COCHINSHIP GRSE DATAPATTNS ZENTEC""",
}
for _sec, _names in _SEC_RAW.items():
    for _n in _names.split():
        SECTORS[_n] = _sec


def load_universe_file():
    """
    A universe.csv / universe.txt committed to the repo wins over everything.
    This is the reliable route to a large universe: NSE blocks datacenter IPs,
    so the live fetch usually fails from GitHub Actions, but YOUR browser can
    download the constituent CSV fine. Accepts either:
        symbol            (one per line, .txt or single-column .csv)
        Symbol,Industry   (the NSE constituent CSV, unmodified)
    """
    for name in ("universe.csv", "universe.txt"):
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            continue
        try:
            if name.endswith(".txt"):
                syms = [l.strip().upper() for l in open(p) if l.strip()]
                sec = {}
            else:
                df = pd.read_csv(p)
                cols = {str(c).strip().lower(): c for c in df.columns}
                scol = cols.get("symbol") or list(df.columns)[0]
                icol = cols.get("industry") or cols.get("sector")
                syms = df[scol].astype(str).str.strip().str.upper().tolist()
                sec = (dict(zip(syms, df[icol].astype(str).str.strip()))
                       if icol else {})
            syms = [x for x in syms if x and x != "NAN"]
            if len(syms) >= 20:
                print(f"universe from {name}: {len(syms)} symbols"
                      f"{' with sectors' if sec else ''}")
                return syms, sec
        except Exception as e:
            print(f"  could not read {name}: {type(e).__name__}")
    return None, {}


def universe():
    syms, sec = load_universe_file()
    if syms:
        return syms, sec
    import io
    try:
        import requests
        hdr = {"User-Agent": "Mozilla/5.0"}
        for u in (
            "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
            "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
            "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        ):
            try:
                r = requests.get(u, headers=hdr, timeout=8)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                sy = df["Symbol"].astype(str).str.strip().str.upper().tolist()
                sc = {}
                if "Industry" in df.columns:
                    sc = dict(zip(sy, df["Industry"].astype(str).str.strip()))
                if len(sy) > 100:
                    print(f"NSE list: {len(sy)} symbols from {u.split('/')[-1]}")
                    return sy, sc
            except Exception as e:
                print(f"  NSE fetch failed ({type(e).__name__})")
    except ImportError:
        pass
    sy = sorted(set(FALLBACK.split()))
    print(f"built-in list: {len(sy)} symbols (commit universe.csv for more)")
    return sy, dict(SECTORS)


def fetch(symbols, batch=40, pause=1.0):
    import yfinance as yf
    panel = {}
    tick = [s + ".NS" for s in symbols]
    for i in range(0, len(tick), batch):
        chunk = tick[i:i + batch]
        print(f"  batch {i//batch+1}/{(len(tick)-1)//batch+1}", flush=True)
        try:
            raw = yf.download(chunk, period=f"{YEARS}y", interval="1d",
                              auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
        except Exception as e:
            print(f"    failed: {type(e).__name__}")
            continue
        for t in chunk:
            try:
                d = raw[t].dropna(subset=["Close"]).copy()
            except Exception:
                continue
            if len(d) < 400 or (d["Close"] <= 0).any():
                continue
            panel[t[:-3]] = add_indicators(d[["Open", "High", "Low", "Close", "Volume"]])
        time.sleep(pause)
    return panel


def fetch_bench():
    """Nifty 50 close series, for relative strength and the regime count."""
    import yfinance as yf
    try:
        b = yf.download(BENCH, period=f"{YEARS}y", interval="1d",
                        auto_adjust=True, progress=False)
        if isinstance(b.columns, pd.MultiIndex):
            b.columns = b.columns.get_level_values(0)
        b = b.dropna(subset=["Close"])
        if len(b) < 300:
            print("  benchmark too short; relative columns will be blank")
            return None
        print(f"benchmark {BENCH}: {len(b)} sessions")
        return b
    except Exception as e:
        print(f"  benchmark fetch failed ({type(e).__name__})")
        return None


def demo_panel(n=90, seed=4):
    """Synthetic panel so the dashboard can be built and reviewed offline."""
    r = np.random.default_rng(seed)
    names = sorted(set(FALLBACK.split()))[:n]
    panel = {}
    for k, sym in enumerate(names):
        rr = np.random.default_rng(seed * 1000 + k)
        px = [100.0]
        px += list(100 * (1 + np.linspace(0, rr.uniform(.3, .9), 300)[1:]))
        px += list(px[-1] * (1 + np.linspace(0, -rr.uniform(.4, .6), 200)[1:]))
        px += list(px[-1] * (1 + np.linspace(0, rr.uniform(.25, .55), 150)[1:]))
        nb = rr.integers(1, 4)
        for _ in range(nb - 1):
            b = px[-1]
            px += list(b * (1 + np.concatenate([np.linspace(0, -.14, 16),
                                                np.linspace(-.14, -.01, 15)])))
            px += list(px[-1] * (1 + np.linspace(0, rr.uniform(.22, .45), 61)[1:]))
        piv, ceil, mb = px[-1], px[-1], []
        pbs = ([.22, .13, .07] if rr.random() < .45 else
               [.07, .13, .22] if rr.random() < .5 else [.14, .09, .18])
        for j, d in enumerate(pbs):
            tr = ceil * (1 - d)
            nx = piv * (1 - .025 - .005 * j)
            mb += list(np.linspace(ceil, tr, 13)) + list(np.linspace(tr, nx, 13))
            ceil = nx
        px += mb
        px = np.array(px) * (1 + rr.normal(0, .004, len(px)))
        L = len(px)
        it = np.abs(rr.normal(0, .006, L))
        vm = np.ones(L)
        vm[-len(mb):] = np.linspace(rr.uniform(.6, 1.6), rr.uniform(.4, 2.4), len(mb))
        idx = pd.bdate_range(end=pd.Timestamp(datetime.now(IST).date()),
                             periods=L + 6)[-L:]
        panel[sym] = add_indicators(pd.DataFrame(
            {"Open": px, "High": px * (1 + it), "Low": px * (1 - it), "Close": px,
             "Volume": (rr.integers(2e6, 6e6, L) * vm).astype(float)}, index=idx))
    return panel


# ----------------------------------------------------------------------
def ret_over(c, lb):
    if len(c) < lb + 1:
        return np.nan
    return float(c.iloc[-1] / c.iloc[-lb - 1] - 1)


def vs_bench(c, bench_c, lb):
    """
    Excess return in percentage points over the benchmark.
    This is the plain-language 'how much stronger than Nifty' number: a value
    of +18 means the stock beat the Nifty by 18 percentage points over the
    window. Distinct from the percentile, which only says where it ranks.
    """
    a, b = ret_over(c, lb), ret_over(bench_c, lb)
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float((a - b) * 100)


def mansfield(c, bench_c, lb):
    """
    Mansfield relative strength: the stock/index ratio measured against its own
    moving average, zero-centred. Above zero = outperforming its recent norm.
    Reads as a trend in leadership rather than a level.
    """
    n = min(len(c), len(bench_c))
    if n < lb + 5:
        return np.nan
    idx = c.index.intersection(bench_c.index)
    if len(idx) < lb + 5:
        return np.nan
    rs = (c.reindex(idx) / bench_c.reindex(idx)).dropna()
    if len(rs) < lb + 5:
        return np.nan
    ma = rs.rolling(lb).mean()
    if not np.isfinite(ma.iloc[-1]) or ma.iloc[-1] == 0:
        return np.nan
    return float((rs.iloc[-1] / ma.iloc[-1] - 1) * 100)


def rs_score_at(df, lb, offset=0):
    """Vol-adjusted RS as it stood `offset` sessions ago."""
    c = df["Close"]
    if offset:
        c = c.iloc[:-offset]
    if len(c) < lb + 5:
        return np.nan
    ret = c.iloc[-1] / c.iloc[-lb] - 1
    vol = c.pct_change().iloc[-lb:].std() * np.sqrt(252)
    return np.nan if (not np.isfinite(vol) or vol <= 0) else ret / vol


def trend_label(delta_pp, band=None):
    band = RS_TREND_BAND if band is None else band
    if not np.isfinite(delta_pp):
        return "—"
    if delta_pp >= band:
        return "RISING"
    if delta_pp <= -band:
        return "FALLING"
    return "FLAT"


def breadth(panel, bench=None, hist=250):
    """
    Market breadth from the panel itself, in three constructions that lag by
    different amounts.

      pct_above_200 / _50  - a LEVEL. Turns the day stocks cross. No lag beyond
                             the moving average itself.
      mcclellan            - 19-EMA minus 39-EMA of the daily net-advance ratio.
                             Differencing two EMAs of the DAILY series, so no
                             cumulation and no integration lag.
      ad_line + ad_ema20   - the cumulative advance-decline line and its 20 EMA.
                             Included for context and divergence, NOT as a
                             trigger: it is an integral, so waiting for the EMA
                             slope to bend stacks two lags and fires well after
                             a trailing stop would have exited.

    Survivorship warning: the panel is today's constituent list, so a cumulative
    AD line computed back over years is biased upward - the losers that were
    delisted are missing. That distortion compounds in a cumulative series and
    barely touches the point-in-time percentages, which is another reason to
    read the levels rather than the line.
    """
    if not panel:
        return None
    close = pd.DataFrame({k: v["Close"] for k, v in panel.items()})
    if close.shape[1] < 20 or len(close) < 260:
        return None

    sma200 = close.rolling(200).mean()
    sma50 = close.rolling(50).mean()
    n200 = (close > sma200).sum(axis=1)
    d200 = sma200.notna().sum(axis=1)
    n50 = (close > sma50).sum(axis=1)
    d50 = sma50.notna().sum(axis=1)
    pct200 = (100.0 * n200 / d200.replace(0, np.nan)).dropna()
    pct50 = (100.0 * n50 / d50.replace(0, np.nan)).dropna()

    chg = close.pct_change()
    adv = (chg > 0).sum(axis=1)
    dec = (chg < 0).sum(axis=1)
    part = (adv + dec).replace(0, np.nan)
    net_ratio = ((adv - dec) / part * 100).dropna()          # daily, not cumulated
    ad_line = (adv - dec).cumsum().astype(float)
    ad_ema = ad_line.ewm(span=20, adjust=False).mean()
    mcc = (net_ratio.ewm(span=19, adjust=False).mean()
           - net_ratio.ewm(span=39, adjust=False).mean())

    hi = close.rolling(252).max()
    lo = close.rolling(252).min()
    nh = (close >= hi).sum(axis=1)
    nl = (close <= lo).sum(axis=1)
    net_hl = (nh - nl).dropna()

    def at(series, back=0):
        if len(series) <= back:
            return np.nan
        return float(series.iloc[-1 - back])

    # Divergence: index at a 60-day high while the AD line is not.
    diverg = None
    if bench is not None:
        bc = bench["Close"]
        idx = bc.index.intersection(ad_line.index)
        if len(idx) > 70:
            b = bc.reindex(idx)
            a = ad_line.reindex(idx)
            b_hi = bool(b.iloc[-1] >= b.iloc[-60:].max())
            a_hi = bool(a.iloc[-1] >= a.iloc[-60:].max())
            diverg = ("negative" if (b_hi and not a_hi)
                      else "positive" if (a_hi and not b_hi)
                      else "none")

    tail = lambda s: [round(float(x), 2) for x in s.iloc[-hist:]]
    dates = [str(d.date()) for d in pct200.index[-hist:]]

    p200, p200_21 = at(pct200), at(pct200, 21)
    slope = (at(ad_ema) - at(ad_ema, 5))
    return dict(
        pct_above_200=round(p200, 1),
        pct_above_200_21d=round(p200_21, 1) if np.isfinite(p200_21) else None,
        pct_above_200_delta=(round(p200 - p200_21, 1)
                             if np.isfinite(p200) and np.isfinite(p200_21) else None),
        pct_above_50=round(at(pct50), 1),
        advances=int(at(adv)), declines=int(at(dec)),
        net_ratio=round(at(net_ratio), 1),
        mcclellan=round(at(mcc), 1),
        mcclellan_21d=round(at(mcc, 21), 1),
        net_new_highs=int(at(net_hl)),
        ad_ema20_slope5=round(slope, 1),
        ad_ema20_falling=bool(slope < 0),
        divergence=diverg,
        state=("EXPANDING" if p200 >= 60 else "MIXED" if p200 >= 40 else "CONTRACTING"),
        series=dict(dates=dates, pct200=tail(pct200),
                    ad=tail(ad_line.reindex(pct200.index).ffill()),
                    ad_ema=tail(ad_ema.reindex(pct200.index).ffill()),
                    mcc=tail(mcc.reindex(pct200.index).ffill())),
    )


def market_regime(bench, panel):
    """
    O'Neil distribution-day count on the Nifty.

    Index volume is unreliable from the vendor, so the volume series is the sum
    of constituent turnover across the panel - the same construction specced
    for the standalone test. A distribution day is the index closing down more
    than DD_DROP on turnover above the prior session, which is institutional
    selling occurring while the index still looks fine.
    """
    if bench is None or not panel:
        return None
    tov = None
    for d in panel.values():
        t = (d["Close"] * d["Volume"]).rename("t")
        tov = t if tov is None else tov.add(t, fill_value=0.0)
    if tov is None:
        return None
    idx = bench.index.intersection(tov.index)
    if len(idx) < DD_WINDOW + 5:
        return None
    c = bench["Close"].reindex(idx)
    v = tov.reindex(idx)
    chg = c.pct_change()
    dd = (chg <= -DD_DROP) & (v > v.shift(1))
    count = int(dd.iloc[-DD_WINDOW:].sum())
    days = [str(d.date()) for d in dd.iloc[-DD_WINDOW:][dd.iloc[-DD_WINDOW:]].index]
    state = ("HEAVY" if count >= 5 else "ELEVATED" if count >= 3 else "LIGHT")
    hi = float(c.iloc[-252:].max()) if len(c) >= 252 else float(c.max())
    return dict(
        distribution_days=count, window=DD_WINDOW, state=state, dates=days,
        index_close=round(float(c.iloc[-1]), 1),
        index_chg=round(float(chg.iloc[-1]) * 100, 2),
        off_high=round((float(c.iloc[-1]) / hi - 1) * 100, 1),
    )


def find_anchor(above, recovery_bars=RECOVERY_BARS):
    """
    Most recent 200-EMA breakdown on the benchmark, ignoring recoveries shorter
    than recovery_bars. That whipsaw guard is what stops the anchor jumping to
    every one-day poke above the EMA.
    """
    run_above, anchor, confirmed = 0, None, True
    for dt, a in above.items():
        if a:
            run_above += 1
            if run_above >= recovery_bars:
                confirmed = True
        else:
            if confirmed:
                anchor = dt
            confirmed = False
            run_above = 0
    return anchor


def anchor_state(bench):
    """Regime + ARS anchor. Returns None if there is no usable benchmark."""
    if bench is None or len(bench) < 60:
        return None
    c = bench["Close"]
    ema200 = c.ewm(span=200, adjust=False).mean()
    above = c > ema200
    if MANUAL_ANCHOR:
        a = pd.Timestamp(MANUAL_ANCHOR)
    else:
        a = find_anchor(above)
        if a is None:
            a = c.index[max(0, len(c) - RS_LONG)]
    a = c.index[c.index.get_indexer([a], method="nearest")[0]]
    return dict(
        anchor=a,
        anchor_date=str(a.date()),
        bars_since=int(len(c.loc[a:])),
        above_200ema=bool(above.iloc[-1]),
        regime=("risk-on" if above.iloc[-1] else "risk-off"),
        nifty_since_anchor=round(float(c.iloc[-1] / c.loc[a] - 1) * 100, 2),
    )


def ars_metrics(s, bench_c, anc):
    """
    Anchored RS for one stock: outperformance of the benchmark measured from the
    anchor, plus the daily ARS series it implies, which gives the streak and the
    improving flag.
    """
    out = dict(ars=np.nan, ars_streak=0, ars_improving=False,
               ret_anchor=np.nan, held_up=False)
    if anc is None or bench_c is None:
        return out
    a = anc["anchor"]
    if a not in s.index or a not in bench_c.index:
        return out
    ret_anchor = float(s.iloc[-1] / s.loc[a] - 1)
    nifty = anc["nifty_since_anchor"] / 100.0
    out["ret_anchor"] = ret_anchor
    out["ars"] = ret_anchor - nifty
    out["held_up"] = bool(ret_anchor > 0 and nifty < 0)

    sc = s.loc[a:] / s.loc[a]
    bc = bench_c.loc[a:] / bench_c.loc[a]
    idx = sc.index.intersection(bc.index)
    if len(idx) < 3:
        return out
    ars_daily = (sc.reindex(idx) / bc.reindex(idx) - 1).dropna()
    if not len(ars_daily):
        return out
    streak = 0
    for v in (ars_daily > 0).iloc[::-1]:
        if v:
            streak += 1
        else:
            break
    out["ars_streak"] = int(streak)
    out["ars_improving"] = bool(
        len(ars_daily) > ARS_IMPROVE_LAG
        and ars_daily.iloc[-1] > ars_daily.iloc[-1 - ARS_IMPROVE_LAG])
    return out


def excess_rs(s, bench_c, lb=RS_LONG):
    """
    Volatility-adjusted EXCESS return over the benchmark, zero-centred.
    RS > 0 means genuinely outperforming the index, which a percentile of
    absolute return cannot express.
    """
    if bench_c is None or len(s) < lb + 1 or len(bench_c) < lb + 1:
        return np.nan
    win = s.iloc[-lb:]
    stock_ret = float(win.iloc[-1] / win.iloc[0] - 1)
    bench_ret = float(bench_c.iloc[-1] / bench_c.iloc[-lb] - 1)
    dvol = float(win.pct_change().std())
    if not np.isfinite(dvol) or dvol <= 0:
        return np.nan
    return (stock_ret - bench_ret) / (dvol * np.sqrt(lb))


def fake_move(s, lb=RS_LONG):
    """
    True when the whole advance is a handful of gaps: if the three biggest up
    days account for more than FAKE_CONC_LIMIT of all up-movement, the trend is
    an artefact rather than accumulation.
    """
    if len(s) < lb + 2:
        return False, np.nan
    win = s.iloc[-lb:]
    rets = win.pct_change().dropna()
    up = rets[rets > 0].sum()
    conc = float(rets.nlargest(3).sum() / up) if up > 0 else 1.0
    stock_ret = float(win.iloc[-1] / win.iloc[0] - 1)
    return bool(stock_ret > 0 and conc > FAKE_CONC_LIMIT), conc


def demo_bench(panel):
    """Synthetic Nifty for demo mode: equal-weight mean of the panel."""
    if not panel:
        return None
    df = pd.DataFrame({k: v["Close"] for k, v in panel.items()}).dropna(how="all")
    c = df.mean(axis=1)
    v = pd.Series(np.random.default_rng(7).integers(2e8, 4e8, len(c)),
                  index=c.index, dtype=float)
    return pd.DataFrame({"Close": c, "Volume": v})


def rs_score(df, lb):
    c = df["Close"]
    if len(c) < lb + 5:
        return np.nan
    ret = c.iloc[-1] / c.iloc[-lb] - 1
    vol = c.pct_change().iloc[-lb:].std() * np.sqrt(252)
    return np.nan if (not np.isfinite(vol) or vol <= 0) else ret / vol


def quadrant(rs, ars):
    """
    The notebook's taxonomy, on signed RS and ARS rather than two percentiles:
      A-LIST   leading over the rolling window AND since the anchor
      AGING    still leads on the rolling window but not since the anchor
      EMERGING lagging on the window but leading since the anchor - the turn
      LAGGARD  neither
    """
    ok_rs = np.isfinite(rs) and rs > 0
    ok_ars = np.isfinite(ars) and ars > 0
    if ok_rs and ok_ars:
        return "A-LIST"
    if ok_rs:
        return "AGING"
    if ok_ars:
        return "EMERGING"
    return "LAGGARD"


def build_rows(panel, bench=None, sectors=None, anc=None):
    sectors = sectors or {}
    bc = bench["Close"] if bench is not None else None

    sl = pd.Series({s: rs_score(d, RS_LONG) for s, d in panel.items()}).dropna()
    ss = pd.Series({s: rs_score(d, RS_SHORT) for s, d in panel.items()}).dropna()
    # Rank descending-safe and cap below 100: with N names a true percentile
    # cannot be 100, and showing 100 for the top few is misleading.
    pl = (sl.rank(pct=True) * 100).clip(upper=99.9).round(1)
    ps = (ss.rank(pct=True) * 100).clip(upper=99.9).round(1)

    # RS percentile as it stood RS_TREND_LOOKBACK sessions ago, so the trend is
    # a genuine cross-sectional move rather than just the stock's own drift.
    slp = pd.Series({s: rs_score_at(d, RS_LONG, RS_TREND_LOOKBACK)
                     for s, d in panel.items()}).dropna()
    plp = (slp.rank(pct=True) * 100).clip(upper=99.9).round(1)

    rows = {}
    for s, d in panel.items():
        res = analyse_stock(d, BASE_CFG["min_len"], BASE_CFG["min_depth"],
                            BASE_CFG["max_depth"], BASE_CFG["min_advance"],
                            BASE_CFG["below_days"], BASE_CFG["dd_thresh"])
        ob = res["open_base"]
        g, checks = stage2_gate(d, len(d) - 1)
        c = d["Close"]
        px = float(c.iloc[-1])
        chg = float(px / c.iloc[-2] - 1) * 100 if len(c) > 1 else 0.0
        tov = float((d["Close"] * d["Volume"]).iloc[-60:].median()) / 1e7
        lp, sp = float(pl.get(s, np.nan)), float(ps.get(s, np.nan))
        lpp = float(plp.get(s, np.nan))
        d_rs = lp - lpp if (np.isfinite(lp) and np.isfinite(lpp)) else np.nan
        v123 = vs_bench(c, bc, RS_LONG) if bc is not None else np.nan
        v55 = vs_bench(c, bc, RS_SHORT) if bc is not None else np.nan
        mans = mansfield(c, bc, RS_LONG) if bc is not None else np.nan

        rs_x = excess_rs(c, bc, RS_LONG)
        A = ars_metrics(c, bc, anc)
        fake, conc = fake_move(c, RS_LONG)
        hi52 = float(c.iloc[-HIGH_LOOKBACK:].max()) if len(c) >= 30 else np.nan
        prox = float(px / hi52) if np.isfinite(hi52) and hi52 > 0 else np.nan
        dma50 = float(c.iloc[-50:].mean()) if len(c) >= 50 else np.nan

        row = dict(symbol=s, price=round(px, 1), chg=round(chg, 2),
                   sector=sectors.get(s) or "Unclassified",
                   rs=None if not np.isfinite(rs_x) else round(rs_x, 3),
                   ars=None if not np.isfinite(A["ars"]) else round(A["ars"] * 100, 1),
                   ars_streak=A["ars_streak"], ars_improving=A["ars_improving"],
                   ret_anchor=(None if not np.isfinite(A["ret_anchor"])
                               else round(A["ret_anchor"] * 100, 1)),
                   held_up=A["held_up"], fake=fake,
                   conc3=None if not np.isfinite(conc) else round(conc, 2),
                   prox52w=None if not np.isfinite(prox) else round(prox, 3),
                   above50=bool(np.isfinite(dma50) and px > dma50),
                   rs_long=None if np.isnan(lp) else lp,
                   rs_short=None if np.isnan(sp) else sp,
                   vs_nifty_long=None if not np.isfinite(v123) else round(v123, 1),
                   vs_nifty_short=None if not np.isfinite(v55) else round(v55, 1),
                   mansfield=None if not np.isfinite(mans) else round(mans, 1),
                   rs_delta=None if not np.isfinite(d_rs) else round(d_rs, 1),
                   rs_trend=trend_label(d_rs),
                   quad=quadrant(rs_x, A["ars"]), stage2=bool(g),
                   stage2_fails=[k for k, v in checks.items() if not v],
                   next_base=int(res["next_base_number"]),
                   count_start=str(res["count_start"].date()),
                   reliable=bool(res["count_reliable"]),
                   turnover_cr=round(tov, 1), in_base=ob is not None,
                   tier=None, hard_fails="", quality_score=None)
        if ob is not None:
            m = base_metrics(res["sub"], ob)
            # A base only a few sessions old cannot be judged on range or volume
            # contraction. Leave tier as None rather than stamping it FAIL for
            # having no history yet.
            q = (quality_flags(m, ob["depth"], QUAL["max_quality_depth"],
                               QUAL["range_max"], QUAL["vol_max"])
                 if m.get("assessable") else None)
            row.update(
                base_start=str(ob["start_date"].date()), base_len=int(ob["length"]),
                depth_pct=round(ob["depth"] * 100, 1),
                pivot=round(ob["peak_val"], 1),
                dist_to_pivot=round(-m["pct_below_pivot"] * 100, 1),
                pullbacks=[round(x, 1) for x in m["pullbacks"]],
                n_pullbacks=int(m["n_pullbacks"]),
                range_contraction=(None if not np.isfinite(m["range_contraction"])
                                   else round(m["range_contraction"], 2)),
                vol_dryup=(None if not np.isfinite(m["vol_dryup"])
                           else round(m["vol_dryup"], 2)),
                pos_in_base=(None if not np.isfinite(m["pos_in_base"])
                             else round(m["pos_in_base"], 2)),
                len_ok=bool(ob["qualifies_len"]),
                depth_ok=bool(ob["qualifies_depth"]))
            if q is not None:
                row.update(tier=q["tier"], hard_fails=q["hard_fails"],
                           quality_score=int(q["quality_score"]),
                           flags={k: bool(q[k]) for k in ALL_FLAGS})
        rows[s] = row
    return rows


def build_lists(rows, anc):
    """
    The four notebook lists, plus the unfiltered RS-positive set.
    Fake movers are excluded from all of them.
    """
    clean = [r for r in rows.values() if not r["fake"]]
    P = lambda r: (r["prox52w"] or 0)
    RS = lambda r: (r["rs"] if r["rs"] is not None else -9e9)
    AR = lambda r: (r["ars"] if r["ars"] is not None else -9e9)

    rolling = sorted([r for r in clean if RS(r) > 0 and P(r) >= PROX_MAIN],
                     key=lambda r: -RS(r))[:TOP_N]
    heldup = sorted([r for r in clean if AR(r) > 0 and P(r) >= PROX_MAIN],
                    key=lambda r: -AR(r))
    rset = {r["symbol"] for r in rolling}
    alist = [r for r in heldup if r["symbol"] in rset]
    alist = sorted(alist, key=lambda r: -RS(r))

    emerging = [r for r in clean
                if RS(r) <= 0 and AR(r) > 0
                and r["ars_streak"] >= ARS_STREAK_MIN
                and r["ars_improving"] and r["above50"]
                and P(r) >= PROX_EMERGING]
    if USE_YOUNG_FILTER:
        emerging = [r for r in emerging if r["ars_streak"] <= YOUNG_STREAK_MAX]
    emerging = sorted(emerging, key=lambda r: -AR(r))

    all_rs_pos = sorted([r for r in clean if RS(r) > 0], key=lambda r: -RS(r))

    return dict(alist=alist, rolling=rolling, heldup=heldup,
                emerging=emerging, rs_pos=all_rs_pos)


def sector_summary(rows, rs_pos):
    """
    Sector counts over the RS-positive set, with average RS and the member names
    - the notebook's view. Share of the sector matters more than the raw count.
    """
    from collections import Counter, defaultdict
    tot = Counter(r["sector"] for r in rows.values())
    grp = defaultdict(list)
    for r in rs_pos:
        grp[r["sector"]].append(r)
    out = []
    for sec, n in tot.items():
        mem = grp.get(sec, [])
        rsv = [m["rs"] for m in mem if m["rs"] is not None]
        arv = [m["ars"] for m in mem if m["ars"] is not None]
        out.append(dict(
            sector=sec, n=n, count=len(mem),
            share=round(100.0 * len(mem) / n, 1) if n else 0.0,
            avg_rs=round(float(np.mean(rsv)), 3) if rsv else None,
            avg_ars=round(float(np.mean(arv)), 1) if arv else None,
            stocks=[m["symbol"] for m in sorted(mem, key=lambda x: -(x["rs"] or 0))]))
    return sorted(out, key=lambda x: (-x["count"], -x["share"]))
    """
    Counts per sector: how many names exist, how many lead on each horizon, and
    the share leading. Share matters more than the raw count - a sector with 8
    names and 5 leaders is doing something a 35-name sector with 6 is not.
    """
    from collections import Counter
    tot = Counter(r["sector"] for r in rows.values())
    ll = Counter(r["sector"] for r in leaders_long)
    ls = Counter(r["sector"] for r in leaders_short)
    med = {}
    for sec in tot:
        vals = [r["vs_nifty_long"] for r in rows.values()
                if r["sector"] == sec and r["vs_nifty_long"] is not None]
        med[sec] = round(float(np.median(vals)), 1) if vals else None
    out = []
    for sec, n in tot.items():
        out.append(dict(sector=sec, n=n, rs_long=ll.get(sec, 0),
                        rs_short=ls.get(sec, 0),
                        share=round(100.0 * ll.get(sec, 0) / n, 1) if n else 0.0,
                        median_vs_nifty=med.get(sec)))
    return sorted(out, key=lambda x: (-x["rs_long"], -x["share"]))


def base_candidates(rows, target=2):
    out = []
    for r in rows.values():
        if not (r["in_base"] and r["stage2"] and r["reliable"]):
            continue
        if r["next_base"] != target:
            continue
        if r.get("base_len", 0) < MIN_FORMING:
            continue
        if not (r.get("len_ok") and r.get("depth_ok")):
            continue
        if r.get("dist_to_pivot", 99) > MAX_PCT_BELOW * 100:
            continue
        if r["turnover_cr"] < MIN_TURNOVER_CR:
            continue
        if (r["rs_long"] or 0) < BASE_RS_MIN:
            continue
        if r.get("tier") is None:          # too young to assess
            continue
        out.append(r)
    order = {"PASS": 0, "NEAR": 1, "FAIL": 2}
    return sorted(out, key=lambda r: (order.get(r["tier"], 3),
                                      -(r["quality_score"] or 0),
                                      -(r["rs_long"] or 0)))


def diff(prev, lists):
    """Annotate each row with what changed since the previous run."""
    prev_lists = (prev or {}).get("lists", {})
    # "new" is a derived view that duplicates rows from the real lists. Letting
    # it into these lookups makes the result depend on dict ordering.
    DERIVED = {"new"}
    real = {k: v for k, v in prev_lists.items() if k not in DERIVED}

    prev_days = {}
    for k, rowsl in real.items():
        for r in rowsl:
            prev_days.setdefault(k, {})[r["symbol"]] = r.get("days_in", 1)

    # Tier is only meaningful for in-base names, so "base" is canonical.
    # setdefault, not assignment, so later lists cannot overwrite it.
    prev_tier = {}
    order_keys = ["base"] + [k for k in real if k != "base"]
    for k in order_keys:
        for r in real.get(k, []):
            if r.get("tier"):
                prev_tier.setdefault(r["symbol"], r["tier"])

    exits = {}
    for k, rowsl in lists.items():
        if k in DERIVED:
            continue
        now = {r["symbol"] for r in rowsl}
        was = set(prev_days.get(k, {}))
        exits[k] = sorted(was - now)
        for r in rowsl:
            s = r["symbol"]
            if s in was:
                r["days_in"] = prev_days[k].get(s, 1) + 1
                r["delta"] = None
            else:
                r["days_in"] = 1
                r["delta"] = "entered"
            # tier movement is tracked SEPARATELY - a name can both enter a
            # list and move tier on the same day, and clobbering one with the
            # other loses information.
            pt = prev_tier.get(s)
            r["tier_delta"] = None
            if r.get("tier") and pt and pt != r["tier"]:
                o = {"FAIL": 0, "NEAR": 1, "PASS": 2}
                r["tier_delta"] = ("tier_up" if o.get(r["tier"], 0) > o.get(pt, 0)
                                   else "tier_down")
                r["tier_from"] = pt
    return exits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data, no network")
    ap.add_argument("--target-base", type=int, default=2)
    a = ap.parse_args()

    os.makedirs(os.path.join(DATA, "history"), exist_ok=True)
    prev = None
    if os.path.exists(LATEST):
        try:
            prev = json.load(open(LATEST))
        except Exception:
            prev = None

    if a.demo:
        print("DEMO MODE - synthetic data")
        panel = demo_panel()
        sectors = dict(SECTORS)
        bench = demo_bench(panel)
    else:
        syms, sectors = universe()
        panel = fetch(syms)
        bench = fetch_bench()
        if not sectors:
            sectors = dict(SECTORS)
    if not panel:
        sys.exit("no data fetched")
    print(f"panel: {len(panel)} symbols")

    anc = anchor_state(bench)
    rows = build_rows(panel, bench, sectors, anc)
    regime = market_regime(bench, panel)
    brd = breadth(panel, bench)
    if regime and anc:
        regime.update(anchor_date=anc["anchor_date"], bars_since=anc["bars_since"],
                      above_200ema=anc["above_200ema"], trend=anc["regime"],
                      nifty_since_anchor=anc["nifty_since_anchor"])
    as_of = max(d.index[-1] for d in panel.values()).date().isoformat()

    L = build_lists(rows, anc)
    base_rows = base_candidates(rows, a.target_base)
    sectors_out = sector_summary(rows, L["rs_pos"])

    lists = {k: [dict(r) for r in v] for k, v in
             (("alist", L["alist"]), ("rolling", L["rolling"]),
              ("heldup", L["heldup"]), ("emerging", L["emerging"]),
              ("rs_pos", L["rs_pos"]), ("base", base_rows))}
    exits = diff(prev, lists)

    new = {}
    for k, rowsl in lists.items():
        for r in rowsl:
            if r.get("delta") == "entered":
                new.setdefault(r["symbol"], dict(r, entered_in=[]))
                new[r["symbol"]]["entered_in"].append(k)
    lists["new"] = sorted(new.values(), key=lambda r: -(r["rs"] or -9e9))

    scan_df = pd.DataFrame([
        dict(symbol=r["symbol"], in_base=r["in_base"], reliable=r["reliable"],
             count_start=r["count_start"], next_base=r["next_base"])
        for r in rows.values()])
    try:
        conf = confound_report(scan_df, panel, verbose=False)
        confound = dict(
            top1=round(conf["top1_year_share"] * 100),
            top2=round(conf["top2_year_share"] * 100),
            hhi=round(conf["monthly_hhi"], 3),
            trough=(None if not np.isfinite(conf["trough_share"] or np.nan)
                    else round(conf["trough_share"] * 100)),
            troughs=[str(t.date()) for t in conf["troughs"]],
            by_year={int(k): int(v) for k, v in conf["by_year"].items()})
        ts = confound["trough"]
        confound["verdict"] = ("no market-wide declines to cluster on" if ts == 0
                              else "confounded with market timing" if (ts or 0) > 60
                              else "partially confounded" if (ts or 0) > 35
                              else "largely idiosyncratic")
    except Exception as e:
        confound = {"verdict": f"unavailable ({type(e).__name__})"}

    tiers = {}
    for r in base_rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    from collections import Counter
    quads = Counter(r["quad"] for r in rows.values())

    out = dict(
        generated_at=datetime.now(IST).isoformat(timespec="seconds"),
        as_of=as_of, demo=bool(a.demo), universe=len(panel),
        target_base=a.target_base,
        params=dict(rs_long=RS_LONG, rs_short=RS_SHORT,
                    leader_pctl=RS_LEADER_PCTL, range_max=QUAL["range_max"],
                    vol_max=QUAL["vol_max"], prox_main=PROX_MAIN,
                    prox_emerging=PROX_EMERGING, top_n=TOP_N,
                    ars_streak_min=ARS_STREAK_MIN),
        counts=dict(alist=len(L["alist"]), rolling=len(L["rolling"]),
                    heldup=len(L["heldup"]), emerging=len(L["emerging"]),
                    rs_pos=len(L["rs_pos"]), new=len(lists["new"]),
                    base=len(base_rows), base_pass=tiers.get("PASS", 0),
                    base_near=tiers.get("NEAR", 0), base_fail=tiers.get("FAIL", 0),
                    fakes=sum(1 for r in rows.values() if r["fake"])),
        quadrants={k: int(v) for k, v in quads.items()},
        exits=exits, confound=confound, regime=regime, breadth=brd,
        sectors=sectors_out,
        base_dist={str(k): int(v) for k, v in
                   pd.Series([r["next_base"] for r in rows.values()
                              if r["in_base"]]).value_counts().sort_index().items()},
        lists=lists)

    with open(LATEST, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    hist = os.path.join(DATA, "history", f"{as_of}.json")
    with open(hist, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"\nas_of {as_of} | universe {len(panel)}")
    if anc:
        print(f"  regime        : {anc['regime']} | anchor {anc['anchor_date']} "
              f"({anc['bars_since']} bars) | Nifty since anchor "
              f"{anc['nifty_since_anchor']:+.2f}%")
    print(f"  quadrants     : " + ", ".join(f"{k} {v}" for k, v in quads.most_common()))
    print(f"  A-LIST        : {len(L['alist'])}")
    print(f"  rolling top{TOP_N}  : {len(L['rolling'])}")
    print(f"  ARS held-up   : {len(L['heldup'])}")
    print(f"  EMERGING      : {len(L['emerging'])}"
          + ("   (zero is informative in a rotation-less tape, not a bug)"
             if not L['emerging'] else ""))
    print(f"  all RS>0      : {len(L['rs_pos'])}")
    print(f"  fakes excluded: {out['counts']['fakes']}")
    print(f"  new today     : {len(lists['new'])}")
    print(f"  base {a.target_base}        : {len(base_rows)} "
          f"(PASS {tiers.get('PASS',0)} / NEAR {tiers.get('NEAR',0)} / "
          f"FAIL {tiers.get('FAIL',0)})")
    print(f"  confound      : {confound.get('verdict')}")
    if brd:
        print(f"  breadth       : {brd['pct_above_200']}% above own 200-DMA "
              f"({brd['pct_above_200_delta']:+.1f} pp in 21d) -> {brd['state']}")
        print(f"  mcclellan     : {brd['mcclellan']}  (21d ago "
              f"{brd['mcclellan_21d']})   net new highs {brd['net_new_highs']:+d}")
        print(f"  AD line       : 20-EMA 5d slope {brd['ad_ema20_slope5']:+.0f}"
              f"  divergence vs index: {brd['divergence']}")
    if regime:
        print(f"  distribution  : {regime['distribution_days']} in "
              f"{regime['window']} sessions -> {regime['state']}"
              f"  (Nifty {regime['off_high']}% off high)")
    print("  top sectors   : " + ", ".join(
        f"{x['sector'].split()[0]} {x['count']}/{x['n']}"
        for x in sectors_out[:5]))
    print(f"wrote {LATEST}")


if __name__ == "__main__":
    main()
