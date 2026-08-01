#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serverless OSINT & Market Intelligence System  —  v2 (Decision Engine)
======================================================================
Correlates physical logistics / military / severe-weather OSINT with the
technical momentum of Copper (HG=F) and Natural Gas (NG=F), then produces a
transparent BUY / SELL / HOLD verdict per commodity.

Outputs on every run:
  1. docs/data.json  -> consumed by the standalone GitHub Pages dashboard
  2. an HTML email    -> Arabic brief + verdicts + an embedded PNG chart

Honesty note: commodity technicals (MACD/RSI) are a genuine directional
signal. OSINT logistics -> price is a WEAK, descriptive correlation. OSINT
therefore only MODULATES confidence; it never flips the technical direction.
All inputs are exposed in data.json so the verdict can be audited/calibrated.

Author: Rabeea  |  Engine: MVC Structure Intelligence
"""

import os
import io
import sys
import ssl
import json
import base64
import smtplib
import datetime as dt
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import feedparser
    import matplotlib
    matplotlib.use("Agg")  # headless backend for CI
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print(f"[FATAL] Dependency import failed: {e}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ASSETS = {
    "Copper": "HG=F",
    "Natural Gas": "NG=F",
}

# Public dashboard URL (GitHub Pages, /docs on main).
DASHBOARD_URL = "https://rabeea-global.github.io/osint-market-intel/"

OUTPUT_DIR = "docs"
DATA_JSON = os.path.join(OUTPUT_DIR, "data.json")

RSS_FEEDS = [
    "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://news.usni.org/feed",
    "https://gcaptain.com/feed/",
    "https://maritime-executive.com/articles.rss",
    "https://www.marinelink.com/news/rss",
    "https://feeds.feedburner.com/defense-update",
]

KEYWORDS = [
    "5th fleet", "b-52", "deployment", "aircraft carrier", "satellite imagery",
    "blockade", "strait", "naval drill", "force majeure", "choke point",
    "lng carrier", "supply chain",
]

# Subset whose presence implies physical SUPPLY DISRUPTION -> upward price risk.
SUPPLY_DISRUPTION = {
    "blockade", "strait", "choke point", "force majeure",
    "lng carrier", "supply chain", "naval drill", "aircraft carrier", "5th fleet",
}

OSINT_LOOKBACK_HOURS = 48
MAX_ALERTS = 15
SERIES_BARS = 60          # trailing bars kept for charting
BUY_THRESHOLD = 0.30      # tech_score gate for a directional call


# ---------------------------------------------------------------------------
# NATIVE INDICATORS
# ---------------------------------------------------------------------------
def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig


def rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# 1. TECHNICAL ANALYSIS + VERDICT
# ---------------------------------------------------------------------------
def analyze_asset(name, ticker):
    r = {
        "name": name, "ticker": ticker, "ok": False, "error": None,
        "close": None, "rsi": None, "macd": None, "macd_signal": None,
        "macd_hist": None, "tech_score": None, "macd_component": None,
        "rsi_component": None, "verdict": "N/A", "verdict_ar": "غير متاح",
        "state": "N/A", "state_ar": "غير متاح",
        "strength": 0, "rationale": "", "rationale_ar": "",
        "series": {"dates": [], "close": [], "macd": [], "signal": [], "rsi": []},
    }
    try:
        df = yf.download(ticker, period="6mo", interval="1d",
                         auto_adjust=False, progress=False, threads=False)
    except Exception as e:
        r["error"] = f"fetch error: {e}"
        return r
    if df is None or df.empty or len(df) < 40:
        r["error"] = "insufficient data"
        return r
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"]).copy()

    try:
        macd_line, signal_line = macd(df["Close"])
        rsi_s = rsi(df["Close"])
    except Exception as e:
        r["error"] = f"indicator error: {e}"
        return r

    df = df.assign(_m=macd_line, _s=signal_line, _r=rsi_s).dropna(subset=["_m", "_s", "_r"])
    if df.empty:
        r["error"] = "no rows after warm-up"
        return r

    last = df.iloc[-1]
    close = float(last["Close"])
    m = float(last["_m"]); s = float(last["_s"]); rv = float(last["_r"])
    hist = m - s

    # scale-free strength: normalise MACD spread by its own recent volatility
    diff = (df["_m"] - df["_s"])
    scale = float(diff.tail(SERIES_BARS).std()) or (abs(diff).tail(SERIES_BARS).mean() or 1.0)
    macd_component = float(np.clip(hist / scale, -1, 1)) if scale else 0.0
    rsi_component = float(np.clip((rv - 50) / 50, -1, 1))
    tech_score = round(0.5 * macd_component + 0.5 * rsi_component, 3)

    bull_gate = (m > s) and (rv < 65)
    bear_gate = (m < s) and (rv > 35)
    if tech_score >= BUY_THRESHOLD and bull_gate:
        verdict, verdict_ar = "BUY", "شراء"
        state, state_ar = "BUY", "شراء"
    elif tech_score <= -BUY_THRESHOLD and bear_gate:
        verdict, verdict_ar = "SELL", "بيع"
        state, state_ar = "SELL", "بيع"
    elif tech_score >= BUY_THRESHOLD and rv >= 65:
        verdict, verdict_ar = "HOLD", "انتظار"
        state, state_ar = "OVERBOUGHT", "تشبّع شرائي — انتظار تصحيح"
    elif tech_score <= -BUY_THRESHOLD and rv <= 35:
        verdict, verdict_ar = "HOLD", "انتظار"
        state, state_ar = "OVERSOLD", "تشبّع بيعي — انتظار ارتداد"
    else:
        verdict, verdict_ar = "HOLD", "انتظار"
        state, state_ar = "NEUTRAL", "محايد — لا أفضلية واضحة"
    strength = int(round(min(1.0, abs(tech_score)) * 100))

    tail = df.tail(SERIES_BARS)
    r.update(
        ok=True, close=round(close, 4), rsi=round(rv, 2),
        macd=round(m, 5), macd_signal=round(s, 5), macd_hist=round(hist, 5),
        tech_score=tech_score, macd_component=round(macd_component, 3),
        rsi_component=round(rsi_component, 3),
        verdict=verdict, verdict_ar=verdict_ar,
        state=state, state_ar=state_ar, strength=strength,
        series={
            "dates": [d.strftime("%Y-%m-%d") for d in tail.index],
            "close": [round(float(x), 4) for x in tail["Close"]],
            "macd": [round(float(x), 5) for x in tail["_m"]],
            "signal": [round(float(x), 5) for x in tail["_s"]],
            "rsi": [round(float(x), 2) for x in tail["_r"]],
        },
    )
    return r


def apply_osint_overlay(asset, supply_pressure):
    """OSINT modulates confidence only; never flips the technical direction."""
    reasons = []
    reasons.append(f"MACD {'>' if asset['macd_hist'] and asset['macd_hist']>0 else '<'} signal")
    reasons.append(f"RSI {asset['rsi']}")
    reasons_ar = [f"MACD {'فوق' if asset['macd_hist'] and asset['macd_hist']>0 else 'تحت'} خط الإشارة",
                  f"RSI {asset['rsi']}"]

    if supply_pressure > 0:
        if asset["verdict"] == "BUY":
            bump = min(15, supply_pressure * 5)
            asset["strength"] = min(95, asset["strength"] + bump)
            reasons.append(f"OSINT supply pressure +{supply_pressure} (raises conviction)")
            reasons_ar.append(f"ضغط لوجستي على العرض +{supply_pressure} (يعزّز القناعة)")
        elif asset["verdict"] == "SELL":
            reasons.append(f"⚠ OSINT supply pressure +{supply_pressure} conflicts with SELL")
            reasons_ar.append(f"⚠ ضغط لوجستي على العرض +{supply_pressure} يتعارض مع إشارة البيع")
        else:
            reasons.append(f"OSINT supply pressure +{supply_pressure} (watch)")
            reasons_ar.append(f"ضغط لوجستي على العرض +{supply_pressure} (مراقبة)")

    asset["rationale"] = "; ".join(reasons)
    asset["rationale_ar"] = "؛ ".join(reasons_ar)
    return asset


# ---------------------------------------------------------------------------
# 2. OSINT MODULE
# ---------------------------------------------------------------------------
def _entry_dt(entry):
    for a in ("published_parsed", "updated_parsed"):
        t = getattr(entry, a, None)
        if t:
            try:
                return dt.datetime(*t[:6])
            except Exception:
                pass
    return None


def scan_osint():
    cutoff = dt.datetime.utcnow() - dt.timedelta(hours=OSINT_LOOKBACK_HOURS)
    alerts, seen = [], set()
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"[WARN] feed failed {url}: {e}", file=sys.stderr)
            continue
        source = feed.feed.get("title", url) if getattr(feed, "feed", None) else url
        for entry in getattr(feed, "entries", []):
            title = (getattr(entry, "title", "") or "").strip()
            summary = (getattr(entry, "summary", "") or "").strip()
            hay = f"{title} {summary}".lower()
            matched = [k for k in KEYWORDS if k in hay]
            if not matched:
                continue
            edt = _entry_dt(entry)
            if edt is not None and edt < cutoff:
                continue
            if title.lower() in seen:
                continue
            seen.add(title.lower())
            supply = any(k in SUPPLY_DISRUPTION for k in matched)
            alerts.append({
                "title": title, "link": getattr(entry, "link", ""),
                "source": source, "keywords": matched,
                "when": edt.strftime("%Y-%m-%d %H:%M") if edt else "—",
                "supply": supply,
            })
    alerts.sort(key=lambda a: a["when"], reverse=True)
    alerts = alerts[:MAX_ALERTS]
    supply_pressure = sum(1 for a in alerts if a["supply"])
    return alerts, supply_pressure


# ---------------------------------------------------------------------------
# 3. OUTPUTS: data.json + chart + email
# ---------------------------------------------------------------------------
def write_data_json(assets, alerts, supply_pressure):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = {
        "generated_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assets": assets,
        "osint": {
            "alert_count": len(alerts),
            "supply_pressure": supply_pressure,
            "alerts": alerts,
        },
    }
    with open(DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {DATA_JSON}")
    return payload


def render_chart_png(assets):
    """3 rows (price, MACD, RSI) x N asset columns. Returns PNG bytes or None."""
    plotable = [a for a in assets if a["ok"] and a["series"]["close"]]
    if not plotable:
        return None
    n = len(plotable)
    plt.rcParams.update({
        "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
        "axes.edgecolor": "#2a2f3a", "text.color": "#c9d1d9",
        "axes.labelcolor": "#c9d1d9", "xtick.color": "#8b949e",
        "ytick.color": "#8b949e", "grid.color": "#20252e", "font.size": 9,
    })
    colors = {"Copper": "#c87941", "Natural Gas": "#3fb8c4"}
    fig, axes = plt.subplots(3, n, figsize=(5.2 * n, 8.0), squeeze=False)
    for j, a in enumerate(plotable):
        c = colors.get(a["name"], "#c9d1d9")
        s = a["series"]; x = range(len(s["close"]))
        # price
        ax = axes[0][j]
        ax.plot(x, s["close"], color=c, lw=1.8)
        ax.set_title(f"{a['name']}  —  {a['verdict']} ({a['strength']}%)",
                     color=c, fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.3); ax.margins(x=0)
        ax.set_ylabel("Price")
        # macd
        ax = axes[1][j]
        ax.plot(x, s["macd"], color=c, lw=1.4, label="MACD")
        ax.plot(x, s["signal"], color="#8b949e", lw=1.2, label="Signal")
        hist = [m - g for m, g in zip(s["macd"], s["signal"])]
        ax.bar(x, hist, color=["#3fb87e" if h >= 0 else "#e05252" for h in hist], alpha=0.5, width=1.0)
        ax.axhline(0, color="#3a3f4a", lw=0.8); ax.grid(True, alpha=0.3); ax.margins(x=0)
        ax.set_ylabel("MACD"); ax.legend(loc="upper left", fontsize=7, framealpha=0.2)
        # rsi
        ax = axes[2][j]
        ax.plot(x, s["rsi"], color=c, lw=1.4)
        ax.axhline(65, color="#e05252", lw=0.8, ls="--")
        ax.axhline(35, color="#3fb87e", lw=0.8, ls="--")
        ax.set_ylim(0, 100); ax.grid(True, alpha=0.3); ax.margins(x=0)
        ax.set_ylabel("RSI")
    fig.suptitle("Copper & Natural Gas — Momentum Snapshot",
                 color="#c9d1d9", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_html_email(payload, chart_cid):
    gen = payload["generated_utc"]
    osint = payload["osint"]
    vcolor = {"BUY": "#3fb87e", "SELL": "#e05252", "HOLD": "#d4a94a", "N/A": "#8b949e"}

    cards = ""
    for a in payload["assets"]:
        col = vcolor.get(a["verdict"], "#8b949e")
        if a["ok"]:
            cards += f"""
            <div style="border:1px solid #2a2f3a;border-radius:10px;padding:14px;margin:8px 0;background:#161b22;">
              <div style="font-size:16px;font-weight:700;color:#c9d1d9;">{a['name']} <span style="color:#8b949e;font-weight:400;">({a['ticker']})</span></div>
              <div style="font-size:26px;font-weight:800;color:{col};margin:6px 0;">{a['verdict_ar']} — {a['verdict']} · قوة {a['strength']}%</div>
              <div style="font-size:13px;color:{col};margin-bottom:4px;">{a['state_ar']}</div>
              <div style="color:#8b949e;font-size:13px;">الإغلاق {a['close']} · RSI {a['rsi']} · MACD {a['macd']} ضد {a['macd_signal']}</div>
              <div style="color:#c9d1d9;font-size:13px;margin-top:6px;">{a['rationale_ar']}</div>
            </div>"""
        else:
            cards += f"""
            <div style="border:1px solid #2a2f3a;border-radius:10px;padding:14px;margin:8px 0;background:#161b22;">
              <div style="font-size:16px;font-weight:700;color:#c9d1d9;">{a['name']}</div>
              <div style="color:#e05252;">تعذّر التحليل: {a['error']}</div>
            </div>"""

    alerts_html = ""
    if osint["alerts"]:
        for al in osint["alerts"]:
            flag = "🔴 عرض" if al["supply"] else "⚪"
            alerts_html += f"""<li style="margin:6px 0;color:#c9d1d9;">{flag} <b>{al['title']}</b>
                <span style="color:#8b949e;">— {al['source']} · {al['when']} · {'، '.join(al['keywords'])}</span></li>"""
    else:
        alerts_html = '<li style="color:#8b949e;">لا توجد تحركات ضمن الكلمات المفتاحية.</li>'

    return f"""<html><body style="background:#0e1117;font-family:Arial,Helvetica,sans-serif;padding:16px;">
      <div style="max-width:720px;margin:auto;">
        <h2 style="color:#c9d1d9;">📊 قرار السوق — النحاس والغاز الطبيعي</h2>
        <div style="color:#8b949e;font-size:12px;">آخر تحديث: {gen} · ضغط العرض (OSINT): {osint['supply_pressure']} من {osint['alert_count']} تنبيه</div>
        {cards}
        <img src="cid:{chart_cid}" style="width:100%;border-radius:10px;border:1px solid #2a2f3a;margin:10px 0;" alt="chart"/>
        <h3 style="color:#c9d1d9;">تنبيهات الحركة اللوجستية (OSINT)</h3>
        <ul style="padding-inline-start:18px;">{alerts_html}</ul>
        <p style="color:#8b949e;font-size:12px;">القرار مبني على الزخم الفني (MACD/RSI)؛ إشارات OSINT تعدّل درجة الثقة فقط ولا تعكس الاتجاه — أداة دعم قرار لا توصية مؤكدة.</p>
        <a href="{DASHBOARD_URL}" style="display:inline-block;background:#c87941;color:#0e1117;padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:700;">افتح لوحة القيادة ↗</a>
      </div>
    </body></html>"""


def send_email(payload):
    email = os.environ.get("EMAIL")
    pw = os.environ.get("APP_PASSWORD")
    if not email or not pw:
        print("[FATAL] EMAIL / APP_PASSWORD not set.", file=sys.stderr)
        sys.exit(1)

    chart = render_chart_png(payload["assets"])
    chart_cid = "momentum_chart"

    msg = MIMEMultipart("related")
    msg["From"] = email
    msg["To"] = email
    verdicts = " · ".join(f"{a['name']} {a['verdict']}" for a in payload['assets'] if a['ok'])
    msg["Subject"] = f"قرار السوق — {verdicts or 'تقرير'} | {dt.datetime.utcnow():%Y-%m-%d}"

    alt = MIMEMultipart("alternative")
    msg.attach(alt)
    # plaintext fallback
    plain = [f"قرار السوق — {payload['generated_utc']}"]
    for a in payload["assets"]:
        if a["ok"]:
            plain.append(f"{a['name']}: {a['verdict']} [{a['state_ar']}] قوة {a['strength']}% — {a['rationale_ar']}")
        else:
            plain.append(f"{a['name']}: تعذّر ({a['error']})")
    plain.append(f"\nالتنبيهات: {payload['osint']['alert_count']} | ضغط العرض: {payload['osint']['supply_pressure']}")
    plain.append(f"\nلوحة القيادة: {DASHBOARD_URL}")
    alt.attach(MIMEText("\n".join(plain), "plain", "utf-8"))
    alt.attach(MIMEText(build_html_email(payload, chart_cid), "html", "utf-8"))

    if chart:
        img = MIMEImage(chart, _subtype="png")
        img.add_header("Content-ID", f"<{chart_cid}>")
        img.add_header("Content-Disposition", "inline", filename="momentum.png")
        msg.attach(img)

    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as srv:
            srv.login(email, pw)
            srv.sendmail(email, email, msg.as_string())
        print("[OK] email sent.")
    except Exception as e:
        print(f"[FATAL] email failed: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=== OSINT & Market Intelligence v2 ===")
    print("[..] OSINT scan")
    alerts, supply_pressure = scan_osint()
    print(f"     {len(alerts)} alerts, supply pressure {supply_pressure}")

    assets = []
    for name, tk in ASSETS.items():
        print(f"[..] {name}")
        a = analyze_asset(name, tk)
        if a["ok"]:
            apply_osint_overlay(a, supply_pressure)
            print(f"     {a['verdict']} [{a['state']}] strength {a['strength']}%")
        else:
            print(f"     FAILED: {a['error']}")
        assets.append(a)

    payload = write_data_json(assets, alerts, supply_pressure)
    print("[..] email")
    send_email(payload)
    print("=== done ===")


if __name__ == "__main__":
    main()
