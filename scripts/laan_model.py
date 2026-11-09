"""Bygger prognosemodellen til Boliglån-appen og tester den bagud.

Skriver laan/model.json med parametre og dataserier. Appen simulerer selv ud fra dagens live-kurser.
Kører i GitHub Actions efter laan_snapshot.py. Kilder: Nasdaq Copenhagen (lukkekurser) og ECB (rentekurve).
Rådata gemmes i laan/data/kilder.json, så en fejlet hentning ikke vælter modellen.

Modeller for 4 % 2059 (alle i kurspoint, uden retning):
  A  Historiske 40-dages forløb (filtreret historisk simulation), skaleret til dagens udsving og centreret.
  B  MA(1)-GARCH(1,1) med t-fordeling (støj i lukkekurser og skiftende udsving).
  K  Kalman: skjult kurs som random walk + støj i lukkekursen (samme modeltype som Codex' efterregning, egne valg).
  C  Rentemodel: ECB 10-årsrente x rentefølsomhed + realkreditspænd. Kun kontrol.
Samlet prognose = A, B og K med faste, lige vægte (ikke tilpasset testen).
5 % 2059 = hældning x 4 %-bevægelse + egen bevægelse (fra 5 %/4 %-parrene 2053 og 2056).

Efterregnet af ChatGPT/Codex i to runder 9.11.2026. Rettet efter runde 1: kædet proxy uden tab ved skift, sigma-indeks,
sidste vindue, ingen datalæk i omregningen fra 2053, tilbudskurs med én dags forsinkelse, centrerede forløb.
Rettet efter runde 2: dansk børskalender i hele kæden, testen på rigtige kalenderfrister, 5 %-vinduer i handelsdage,
centrering efter udtynding, betinget Kalman-start, konvergens foretrækkes, samme C-opskrift live og i test.
Stadig åbent: testen måler Nasdaq-lukkekurser, mens appen viser Totalkredits aktuelle kurs; 5 %-åbning kan ikke testes.
"""
import csv
import io
import json
import math
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "laan" / "data" / "kilder.json"
OUT = ROOT / "laan" / "model.json"
HIST = ROOT / "laan" / "history.json"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) laan-model"}
BONDS = {"4_59": "TX6851322", "5_59": "TX7645240", "4_56": "TX4766950", "5_56": "TX4766962",
         "4_53": "TX4399546", "5_53": "TX4450636"}
NASDAQ = "https://api.nasdaq.com/api/nordic/instruments/{id}/chart?assetClass=MORTGAGE_BONDS&fromDate=2022-01-01&toDate={til}&lang=en"
ECB = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y?format=csvdata&startPeriod=2005-01-01"
H = 40            # simulerede handelsdage (8 uger er højst 40)
LAM = 0.97        # EWMA-hukommelse for udsving
WARM = 60         # EWMA-opvarmning: vinduer før dette indeks bruges ikke
SPLICE_59 = "2026-04-01"   # 4 % 2059 bruges selv fra april (marts var åbningsuger)
K_OBS = 150       # Kalman fittes på de seneste 150 dagsændringer


def paaske(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e, f = b // 4, b % 4, (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, kk = c // 4, c % 4
    ll = (32 + 2 * e + 2 * i - h - kk) % 7
    m = (a + 11 * h + 22 * ll) // 451
    return date(y, (h + ll - 7 * m + 114) // 31, ((h + ll - 7 * m + 114) % 31) + 1)


def lukkedage(y):
    """Lukkedage på Nasdaq Copenhagen (ud over weekender). Passer med alle kursdata 2022-2026."""
    p = paaske(y)
    s = {date(y, 1, 1), p - timedelta(3), p - timedelta(2), p + timedelta(1), p + timedelta(39), p + timedelta(40),
         p + timedelta(50), date(y, 6, 5), date(y, 12, 24), date(y, 12, 25), date(y, 12, 26), date(y, 12, 31)}
    if y <= 2023:
        s.add(p + timedelta(26))      # store bededag (afskaffet fra 2024)
    return s


def handelsdage(fra, til):
    hol = set().union(*[lukkedage(y) for y in range(fra.year, til.year + 1)])
    out, d = [], fra
    while d <= til:
        if d.weekday() < 5 and d not in hol:
            out.append(d.isoformat())
        d += timedelta(1)
    return out


def hent(url, as_json=True):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        data = r.read()
    return json.loads(data) if as_json else data.decode()


def opdater_kilder():
    kilder = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    til = date.today().isoformat()
    for navn, oid in BONDS.items():
        try:
            cp = hent(NASDAQ.format(id=oid, til=til))["data"]["CP"]
            s = kilder.setdefault(navn, {})
            for p in cp:
                s[p["z"]["dateTime"]] = float(p["z"]["close"])
        except Exception as ex:  # Nasdaq blokerer af og til cloud-IP'er: brug cache
            print("Nasdaq fejlede for", navn, ex)
    try:
        rows = list(csv.reader(io.StringIO(hent(ECB, as_json=False))))
        i, j = rows[0].index("TIME_PERIOD"), rows[0].index("OBS_VALUE")
        s = kilder.setdefault("y10", {})
        for r in rows[1:]:
            if r[j]:
                s[r[i]] = float(r[j])
    except Exception as ex:
        print("ECB fejlede", ex)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({k: dict(sorted(v.items())) for k, v in kilder.items()}, separators=(",", ":")))
    return kilder


def hbeta(a, b, h, til=None):
    """Hældning af b på a over h-observations ændringer (fælles datoer, evt. kun til og med 'til')."""
    ds = sorted(set(a) & set(b))
    if til:
        ds = [d for d in ds if d <= til]
    A, B = np.array([a[d] for d in ds]), np.array([b[d] for d in ds])
    x, y = A[h:] - A[:-h], B[h:] - B[:-h]
    bb = float(x @ y / (x @ x))
    return bb, float(np.std(y - bb * x)), len(x), float(np.corrcoef(x, y)[0, 1])


def byg_proxy(k):
    """Kædet dagsserie for 4 % 2059. For hvert par af handelsdage bruges den bedste obligation med kurs begge dage:
    2059 (fra april 2026) > 2056 > 2053 (skaleres med b53, som estimeres uden datalæk i testen)."""
    s53, s56, s59 = k["4_53"], k["4_56"], k["4_59"]
    obs = set(s53) | set(s56) | set(s59)
    alle = sorted(obs)
    cal = handelsdage(date.fromisoformat(alle[0]), date.fromisoformat(alle[-1]))
    udenfor = obs - set(cal)
    if udenfor:
        raise ValueError(f"Kurser på lukkedage: {sorted(udenfor)[:5]}")
    dp, src, huller = [], [], []
    have = [d for d in cal if d in obs]
    pos = {d: i for i, d in enumerate(cal)}
    for a, b in zip(have, have[1:]):
        if b >= SPLICE_59 and a in s59 and b in s59:
            x, s = s59[b] - s59[a], 2
        elif a in s56 and b in s56:
            x, s = s56[b] - s56[a], 1
        elif a in s53 and b in s53:
            x, s = s53[b] - s53[a], 0       # skaleres senere
        else:
            raise ValueError(f"Hul i kæden {a}->{b}")
        n = pos[b] - pos[a]                  # handelsdage mellem observationerne
        if n > 1:
            huller.append(a)
        dp += [x / n] * n; src += [s] * n
    return cal, np.array(dp), np.array(src), huller


def skaler53(dp, src, b53):
    out = dp.copy()
    out[src == 0] *= b53
    return out


def ewma_sig(dp, lam=LAM, init=WARM):
    """sig[i] = udsving kendt ved niveau i (efter dp[0..i-1]); sig har længde len(dp)+1."""
    v = float(np.mean(dp[:init] ** 2))
    sig = np.empty(len(dp) + 1)
    for i, x in enumerate(dp):
        sig[i] = math.sqrt(v)
        v = lam * v + (1 - lam) * x * x
    sig[-1] = math.sqrt(v)
    return sig


# ---------- Model B: MA(1)-GARCH(1,1)-t ----------
def garch_nll(p, x):
    th, w, a, b, nu = p
    if not (-0.9 < th < 0.9 and w > 0 and a >= 0 and b >= 0 and a + b < 0.999 and 2.1 < nu < 60):
        return 1e10
    e_prev, v = 0.0, float(np.var(x))
    ll = 0.0
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    for t in range(len(x)):
        e = x[t] - th * e_prev
        ll += c - 0.5 * math.log(v) - (nu + 1) / 2 * math.log(1 + e * e / (v * (nu - 2)))
        e_prev = e
        v = w + a * e * e + b * v
    return -ll


def fit_garch(x):
    v = float(np.var(x))
    best = None
    for th0, nu0 in ((-0.15, 6.0), (-0.05, 4.5), (-0.25, 12.0)):
        r = minimize(garch_nll, [th0, v * 0.05, 0.08, 0.87, nu0], args=(x,), method="Nelder-Mead",
                     options={"maxiter": 6000, "xatol": 1e-7, "fatol": 1e-5})
        if r.fun < 1e9 and (best is None or (r.success, -r.fun) > (best.success, -best.fun)):
            best = r                     # konvergerede fit foretrækkes, derefter laveste -loglik
    if best is None:
        raise RuntimeError("GARCH-fit fejlede")
    th, w, a, b, nu = best.x
    e_prev, vv = 0.0, v
    for t in range(len(x)):
        e = x[t] - th * e_prev
        e_prev = e
        vv = w + a * e * e + b * vv
    return {"theta": th, "omega": w, "alpha": a, "beta": b, "nu": nu, "v_next": vv, "e_last": e_prev,
            "nll": best.fun, "ok": bool(best.success)}


def sim_garch(g, n, rng):
    nu = g["nu"]
    sc = math.sqrt((nu - 2) / nu)
    v = np.full(n, g["v_next"])
    e_prev = np.full(n, g["e_last"])
    paths = np.empty((n, H))
    cum = np.zeros(n)
    for d in range(H):
        e = np.sqrt(v) * rng.standard_t(nu, n) * sc
        cum += e + g["theta"] * e_prev
        paths[:, d] = cum
        e_prev = e
        v = g["omega"] + g["alpha"] * e * e + g["beta"] * v
    return paths


# ---------- Model K: Kalman (skjult kurs + støj) ----------
def kalman_nll(lp, y):
    q, r = math.exp(lp[0]), math.exp(lp[1])
    m, P = y[0], r                       # betinget på første observation
    ll = 0.0
    for t in range(1, len(y)):
        P += q
        S = P + r
        e = y[t] - m
        ll += -0.5 * (math.log(2 * math.pi * S) + e * e / S)
        K = P / S
        m += K * e
        P *= (1 - K)
    return -ll


def fit_kalman(levels):
    y = np.asarray(levels, float)
    best = None
    for q0, r0 in ((0.08, 0.01), (0.03, 0.05), (0.2, 0.001)):
        r = minimize(kalman_nll, [math.log(q0), math.log(r0)], args=(y,), method="Nelder-Mead",
                     options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or (r.success, -r.fun) > (best.success, -best.fun):
            best = r
    q, rr = math.exp(best.x[0]), math.exp(best.x[1])
    m, P = y[0], rr
    for t in range(1, len(y)):
        P += q; K = P / (P + rr); m += K * (y[t] - m); P *= (1 - K)
    return {"q": q, "r": rr, "m_last": m, "P_last": P, "y_last": float(y[-1]), "ok": bool(best.success)}


def sim_kalman_obs(kf, n, rng):
    """Forløb af fremtidige LUKKEKURSER minus sidste lukkekurs (til testen bagud)."""
    start = (kf["m_last"] - kf["y_last"]) + rng.standard_normal(n) * math.sqrt(kf["P_last"])
    lat = start[:, None] + np.cumsum(rng.standard_normal((n, H)) * math.sqrt(kf["q"]), axis=1)
    return lat + rng.standard_normal((n, H)) * math.sqrt(kf["r"])


# ---------- Model A og C: historiske forløb ----------
def vinduer(levels, starts):
    return np.stack([levels[s + 1:s + H + 1] - levels[s] for s in starts])


def sim_fhs(levels, sig, t, rng, n):
    """Vinduer helt inden for niveau 0..t; skaleret til udsvinget kendt ved t; centreret (ingen retning)."""
    starts = np.arange(WARM, t - H + 1)
    if len(starts) < 20:
        return None
    pick = rng.choice(starts, n)
    W = vinduer(levels, pick) * (sig[t] / sig[pick])[:, None]
    return W - W.mean(0)


def crps(sample, y):
    s = np.sort(sample)
    n = len(s)
    return float(np.mean(np.abs(s - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / (n * n))


# ---------- 5 %-modellen ----------
def model_5(k, hist, cal):
    """Hældning og egen bevægelse for 5 % i forhold til 4 %. Alle ændringer matchet på datoer."""
    calset = {d: i for i, d in enumerate(cal)}
    tk = {r["date"]: (r["k4_aktuel"], r["k5_aktuel"]) for r in hist if r.get("k4_aktuel") and r.get("k5_aktuel")}
    # Totalkredit aktuel kurs: 5 handelsdages ændringer, datomatchet via børskalenderen
    xs, ys = [], []
    for d, (a4, a5) in tk.items():
        if d not in calset or calset[d] + 5 >= len(cal):
            continue
        d2 = cal[calset[d] + 5]
        if d2 in tk:
            xs.append(tk[d2][0] - a4); ys.append(tk[d2][1] - a5)
    b_tk = r_tk = None
    if len(xs) >= 8:
        x, y = np.array(xs), np.array(ys)
        b_tk = float(x @ y / (x @ x)); r_tk = float(np.std(y - b_tk * x))
    # Analoge par: 40-dages ændringer fra starter med 5 % i 99,5-102,5 (samme vinduer som egenbevægelsen)
    # Analoge par på børskalenderen: H handelsdage; start og slut skal være handlet, højst 5 manglende dage undervejs
    an_b, an_r, eta = [], [], []
    for a5, a4 in (("5_53", "4_53"), ("5_56", "4_56")):
        ds0 = sorted(set(k[a5]) & set(k[a4]))
        dcal = handelsdage(date.fromisoformat(ds0[0]), date.fromisoformat(ds0[-1]))
        obs = np.array([d in k[a5] and d in k[a4] for d in dcal])
        ff = lambda s: np.array(list(__import__("itertools").accumulate([s.get(d) for d in dcal], lambda a, b: b if b is not None else a)), float)
        p5, p4 = ff(k[a5]), ff(k[a4])
        mangler = np.concatenate([[0], np.cumsum(~obs)])
        starts = [i for i in range(len(dcal) - H) if obs[i] and obs[i + H] and 99.5 <= p5[i] < 102.5
                  and mangler[i + H + 1] - mangler[i] <= 5]
        x = np.array([p4[i + H] - p4[i] for i in starts]); y = np.array([p5[i + H] - p5[i] for i in starts])
        x0, y0 = x - x.mean(), y - y.mean()
        bb = float(x0 @ y0 / (x0 @ x0)); an_b.append(bb)
        lvl = p5 - bb * p4
        W = np.stack([lvl[s + 1:s + H + 1] - lvl[s] for s in starts])[::3]   # hvert 3. vindue (filstørrelse)
        W -= W.mean(0)                                   # centreret efter udtynding: ingen indbygget retning
        s5 = [i for i in starts if obs[i + 5]]
        x5 = np.array([p4[i + 5] - p4[i] for i in s5]); y5 = np.array([p5[i + 5] - p5[i] for i in s5])
        an_r.append(float(np.std((y5 - y5.mean()) - bb * (x5 - x5.mean()))))
        corr40 = float(np.corrcoef(W[:, -1], x[::3])[0, 1])
        eta.append({"W": [[round(float(v), 3) for v in w] for w in W], "n": len(starts), "corr40": corr40})
    b_an = float(np.mean(an_b))
    b5 = float(np.mean([v for v in (b_tk, b_an) if v is not None]))
    raw_scale = None if r_tk is None else r_tk / float(np.mean(an_r))
    eta_scale = 0.4 if raw_scale is None else float(max(raw_scale, 0.4))   # gulv 0,4 = forsigtighedsantagelse
    # Niveauregression (kun til orientering)
    lv = np.array([v for v in tk.values()])
    b_lvl = float(np.polyfit(lv[:, 0], lv[:, 1], 1)[0]) if len(lv) >= 10 else None
    return {"b5": b5, "b_tk": b_tk, "b_analog": an_b, "b_lvl_info": b_lvl, "eta_scale": eta_scale,
            "eta_scale_raw": raw_scale, "r_tk": r_tk, "r_analog": an_r, "eta": eta, "n_tk": len(xs)}


def spaend(hist, cal, kk, cap=None):
    """Tilbudskurs(t) - aktuel kurs(forrige børsdag), matchet på børskalenderen."""
    by = {r["date"]: r for r in hist}
    calset = {d: i for i, d in enumerate(cal)}
    v = []
    for d, r in sorted(by.items()):
        if not r.get(kk + "_tilbud") or (cap and r[kk + "_tilbud"] >= cap) or d not in calset or calset[d] == 0:
            continue
        p = by.get(cal[calset[d] - 1])
        if p and p.get(kk + "_aktuel"):
            v.append(r[kk + "_tilbud"] - p[kk + "_aktuel"])
    if len(v) < 5:
        return None, None, len(v)
    return float(np.mean(v[-20:])), float(np.std(v[-20:])), len(v)


def main():
    k = opdater_kilder()
    hist = json.loads(HIST.read_text())
    cal, dp_raw, src, huller = byg_proxy(k)
    # Børskalender til Totalkredit-matching (frem til seneste Totalkredit-dato)
    cal_tk = handelsdage(date.fromisoformat(cal[0]), max(date.fromisoformat(cal[-1]), date.fromisoformat(hist[-1]["date"])))
    b53 = hbeta(k["4_53"], k["4_56"], 10)[0]
    dp = skaler53(dp_raw, src, b53)
    P = np.concatenate([[0.0], np.cumsum(dp)])
    sig = ewma_sig(dp)
    n = len(dp)
    sig_lr = float(np.sqrt(np.mean(dp ** 2)))

    ydates = sorted(k["y10"])
    Y = np.array([k["y10"][d] for d in ydates])
    dy = np.diff(Y)
    ysig = ewma_sig(dy)

    s59 = {d: v for d, v in k["4_59"].items() if d >= SPLICE_59}
    D59, res59, nD, cD = hbeta(k["y10"], s59, 10)
    D = -D59 if nD >= 60 else -hbeta(k["y10"], k["4_56"], 10)[0]
    # Model C (kontrol): samme opskrift som i testen bagud: D og spænd fra de seneste 250 fælles datoer
    pid = [i for i, d in enumerate(cal) if d in k["y10"]][-250:]
    pp = P[pid]; yy = np.array([k["y10"][cal[i]] for i in pid])
    xx, yv = yy[10:] - yy[:-10], pp[10:] - pp[:-10]
    D_c = -float(xx @ yv / (xx @ xx))
    R = pp + D_c * yy

    k5 = model_5(k, hist, cal_tk)
    s4, s4sd, ns4 = spaend(hist, cal_tk, "k4")
    s5, s5sd, ns5 = spaend(hist, cal_tk, "k5", cap=101.1)
    if s5 is None:
        s5, s5sd = s4, s4sd

    g = fit_garch(dp)
    kf = fit_kalman(P[-(K_OBS + 1):])

    # Totalkredit vs Nasdaq: udsving på samme datoer (målvariablen er Totalkredits aktuelle kurs)
    tkd = [(r["date"], r["k4_aktuel"]) for r in hist if r.get("k4_aktuel")]
    pairs = [(tkd[i][1] - tkd[i - 1][1], k["4_59"][tkd[i][0]] - k["4_59"][tkd[i - 1][0]])
             for i in range(1, len(tkd)) if tkd[i][0] in k["4_59"] and tkd[i - 1][0] in k["4_59"]]
    tk_vs_nq = None
    if len(pairs) >= 8:
        a = np.array(pairs)
        tk_vs_nq = {"n": len(a), "sd_tk": float(a[:, 0].std()), "sd_nq": float(a[:, 1].std()),
                    "corr": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])}

    bt = backtest(k, cal, dp_raw, src, ydates, Y)

    sl = np.sort(sig[WARM:])
    model = {
        "opdateret": datetime.now(ZoneInfo("Europe/Copenhagen")).isoformat(timespec="minutes"),
        "data": {"proxy_fra": cal[0], "proxy_til": cal[-1], "n_dage": n, "b53": round(b53, 3),
                 "kilder": {"2053": int((src == 0).sum()), "2056": int((src == 1).sum()), "2059": int((src == 2).sum())},
                 "huller": huller,
                 "lukket": sorted(d.isoformat() for y in (date.today().year, date.today().year + 1) for d in lukkedage(y)
                                  if d.weekday() < 5 and d >= date.today()),
                 "y10_fra": ydates[0], "y10_til": ydates[-1], "y10_nu": float(Y[-1]), "tk_vs_nq": tk_vs_nq},
        "A": {"dp": [round(float(x), 4) for x in dp], "sig_now": float(sig[-1]), "sig_lr": sig_lr,
              "sig_pct": float(np.searchsorted(sl, sig[-1]) / len(sl)), "warm": WARM, "lam": LAM},
        "B": g,
        "K": kf,
        "C": {"dy": [round(float(x), 5) for x in dy], "ysig_now": float(ysig[-1]),
              "ysig_lr": float(np.sqrt(np.mean(dy ** 2))), "D": D, "D_corr": -cD, "D_n": nD, "D_c": D_c,
              "R": [round(float(x), 4) for x in R]},
        "k5": k5,
        "spaend": {"s4": s4, "s4_sd": s4sd, "s5": s5, "s5_sd": s5sd, "n4": ns4, "n5": ns5},
        "test": bt,
    }
    OUT.write_text(json.dumps(model, separators=(",", ":")))
    print(json.dumps({kk: v for kk, v in model.items() if kk not in ("A", "C", "k5", "test")}, indent=1, default=str))
    print("k5", {kk: v for kk, v in k5.items() if kk != "eta"}, [e["corr40"] for e in k5["eta"]])
    for m, v in bt["modeller"].items():
        print(m, {h: (round(v[h]["cov80"], 2), round(v[h]["crps"], 3), v[h]["n"]) for h in ("1", "4", "8") if h in v},
              {e: (round(x["p"], 2), round(x["o"], 2), round(x["brier"], 3)) for e, x in v.get("touch", {}).items()})
    print("bootstrap", json.dumps(bt["bootstrap"]))


def backtest(k, cal, dp_raw, src, ydates, Y, step=5, n=2000):
    """Rullende test: for hvert startpunkt bruges kun data til og med startdagen (også omregningen fra 2053).
    Horisonter er rigtige kalenderfrister (+1, +4 og +8 uger), talt i handelsdage på børskalenderen."""
    rng = np.random.default_rng(7)
    first = 500
    datoer = [date.fromisoformat(d) for d in cal]
    UGER = (1, 4, 8)
    def frist(t, w):                              # antal handelsdage fra start t til og med dato + 7w
        slut = datoer[t] + timedelta(7 * w)
        j = t
        while j + 1 < len(datoer) and datoer[j + 1] <= slut:
            j += 1
        return j - t if datoer[t] + timedelta(7 * w) <= datoer[-1] else None
    origins = list(range(first, len(dp_raw) - 4, step))
    yi = {d: i for i, d in enumerate(ydates)}
    EV = {"op1_4u": (1.0, 4, True), "op1_8u": (1.0, 8, True), "ned15_8u": (1.5, 8, False)}
    MODS = ("naiv", "lokal", "A", "B", "K", "C", "ABK")
    res = {m: {w: {"c50": [], "c80": [], "c90": [], "crps": []} for w in UGER} for m in MODS}
    touch = {m: {e: [] for e in EV} for m in MODS}
    per_origin = {m: {w: [] for w in UGER} for m in MODS}    # (dækning 80 %, CRPS) til blok-bootstrap
    g = None
    for j, t in enumerate(origins):
        d0 = cal[t]                              # startdag = niveau t
        b53_t = hbeta(k["4_53"], k["4_56"], 10, til=d0)[0]
        dp = skaler53(dp_raw[:t], src[:t], b53_t)
        dp_all = skaler53(dp_raw, src, b53_t)    # fremtiden bruges kun til at måle udfaldet
        P = np.concatenate([[0.0], np.cumsum(dp_all)])
        sig = ewma_sig(dp)                       # længde t+1; sig[t] = kendt ved start
        sl = float(np.sqrt(np.mean(dp ** 2)))
        sims = {}
        sims["naiv"] = np.cumsum(rng.standard_normal((n, H)) * sl, axis=1)
        sims["lokal"] = np.cumsum(rng.standard_normal((n, H)) * sig[t], axis=1)
        sims["A"] = sim_fhs(P[:t + 1], sig, t, rng, n)
        if g is None or j % 12 == 0:
            g = fit_garch(dp)
        th, w_, a, b = g["theta"], g["omega"], g["alpha"], g["beta"]
        e_prev, vv = 0.0, float(np.var(dp))
        for x in dp:
            e = x - th * e_prev; e_prev = e; vv = w_ + a * e * e + b * vv
        sims["B"] = sim_garch(dict(g, v_next=vv, e_last=e_prev), n, rng)
        sims["K"] = sim_kalman_obs(fit_kalman(P[t - K_OBS:t + 1]), n, rng)
        ty = max((i for d, i in yi.items() if d <= d0), default=None)
        pid = [i for i in range(t + 1) if cal[i] in yi][-250:]
        if ty and ty > 500 and len(pid) > 120:
            pp = P[pid]; yy = np.array([Y[yi[cal[i]]] for i in pid])
            x, yv = yy[10:] - yy[:-10], pp[10:] - pp[:-10]
            Dt = -float(x @ yv / (x @ x))
            Rr = pp + Dt * yy
            ys = ewma_sig(np.diff(Y[:ty + 1]))
            pick = rng.choice(np.arange(WARM, ty - H + 1), n)
            Wy = vinduer(Y, pick) * (ys[ty] / ys[pick])[:, None]
            Wr = vinduer(Rr, rng.choice(np.arange(0, len(Rr) - H), n))
            S = -Dt * Wy + Wr
            sims["C"] = S - S.mean(0)
        else:
            sims["C"] = None
        av = [sims[m] for m in ("A", "B", "K") if sims[m] is not None]
        sims["ABK"] = np.concatenate(av)[rng.permutation(n * len(av))[:n]]
        hw = {w: frist(t, w) for w in UGER}
        for m, S in sims.items():
            if S is None:
                continue
            for w in UGER:
                h = hw[w]
                if h is None or t + h >= len(P):
                    continue
                yobs = P[t + h] - P[t]
                col = S[:, h - 1]
                q = np.quantile(col, [0.05, 0.1, 0.25, 0.75, 0.9, 0.95])
                r = res[m][w]
                r["c50"].append(q[2] <= yobs <= q[3]); r["c80"].append(q[1] <= yobs <= q[4]); r["c90"].append(q[0] <= yobs <= q[5])
                r["crps"].append(crps(col[:500], yobs))
                per_origin[m][w].append((t, float(q[1] <= yobs <= q[4]), r["crps"][-1]))
            for e, (kk, w, up) in EV.items():
                h = hw[w]
                if h is None or t + h >= len(P):
                    continue
                path = P[t + 1:t + h + 1] - P[t]
                pr = float(np.mean(S[:, :h].max(1) >= kk)) if up else float(np.mean(S[:, :h].min(1) <= -kk))
                ob = float(path.max() >= kk) if up else float(path.min() <= -kk)
                touch[m][e].append((pr, ob))
    out = {"fra": cal[origins[0]], "til": cal[origins[-1]], "n_start": len(origins), "step": step, "modeller": {}}
    for m in MODS:
        mm = {}
        for w in UGER:
            r = res[m][w]
            if r["crps"]:
                mm[str(w)] = {"cov50": float(np.mean(r["c50"])), "cov80": float(np.mean(r["c80"])),
                              "cov90": float(np.mean(r["c90"])), "crps": float(np.mean(r["crps"])), "n": len(r["crps"])}
        mm["touch"] = {}
        for e in EV:
            tt = np.array(touch[m][e])
            if len(tt):
                mm["touch"][e] = {"p": float(tt[:, 0].mean()), "o": float(tt[:, 1].mean()),
                                  "brier": float(np.mean((tt[:, 0] - tt[:, 1]) ** 2)), "n": len(tt)}
        out["modeller"][m] = mm
    # Parret blok-bootstrap (blokke på 8 starter ~ 8 uger): usikkerhed på dækning og på fejl i forhold til de to simple modeller
    brng = np.random.default_rng(11)
    bs = {}
    for m in ("lokal", "A", "B", "K", "C", "ABK"):
        bm = {}
        for w in UGER:
            a = {x[0]: x for x in per_origin[m][w]}
            nv = {x[0]: x for x in per_origin["naiv"][w]}
            lk = {x[0]: x for x in per_origin["lokal"][w]}
            ts = [t for t in a if t in nv and t in lk]
            if len(ts) < 16:
                continue
            A_ = np.array([a[t][1:] for t in ts]); N_ = np.array([nv[t][2] for t in ts]); L_ = np.array([lk[t][2] for t in ts])
            Lb, nb = 8, int(math.ceil(len(ts) / 8))
            cov, sn, slk = [], [], []
            for _ in range(2000):
                idx = np.concatenate([np.arange(s, s + Lb) for s in brng.integers(0, len(ts) - Lb + 1, nb)])[:len(ts)]
                cov.append(A_[idx, 0].mean())
                sn.append(1 - A_[idx, 1].mean() / N_[idx].mean()); slk.append(1 - A_[idx, 1].mean() / L_[idx].mean())
            q = lambda v: [float(np.quantile(v, .05)), float(np.quantile(v, .95))]
            bm[str(w)] = {"cov80": q(cov), "skill_naiv": q(sn), "skill_lokal": q(slk),
                          "skill_lokal_pkt": float(1 - A_[:, 1].mean() / L_.mean())}
        bs[m] = bm
    out["bootstrap"] = bs
    return out


if __name__ == "__main__":
    main()
