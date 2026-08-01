#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serverless OSINT & Market Intelligence System
==============================================
Correlates physical logistical / military / severe-weather signals (OSINT)
with the technical momentum of Copper (HG=F) and Natural Gas (NG=F).

Runs headless on a GitHub Actions runner and self-mails an Arabic strategic
brief via Gmail SMTP (SSL 465).

Author: Rabeea  |  Engine: MVC Structure Intelligence
"""

import os
import sys
import ssl
import smtplib
import datetime as dt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Third-party
# NOTE: Indicators are computed natively (pandas ewm) instead of pandas_ta.
# pandas_ta's current line (0.4.x) hard-pins numba==0.61.2 and requires
# pandas>=2.3.2 / numpy>=2.2.6 — heavy and volatile for an unattended cron.
# Native MACD/RSI match pandas_ta to ~1e-7 at the evaluated bar (verified).
try:
    import yfinance as yf
    import pandas as pd
    import feedparser
except Exception as e:  # pragma: no cover
    print(f"[FATAL] Dependency import failed: {e}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# NATIVE INDICATORS (no heavy TA dependency)
# ---------------------------------------------------------------------------
def macd(close: "pd.Series", fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line) using classic EMA definition."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def rsi(close: "pd.Series", length: int = 14):
    """Wilder's RSI (RMA smoothing). Matches pandas_ta to floating precision."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ASSETS = {
    "Copper (HG=F)": "HG=F",
    "Natural Gas (NG=F)": "NG=F",
}

# Specialized OSINT feeds: defense, maritime, strategic tracking.
# feedparser degrades gracefully on any dead/relocated feed.
RSS_FEEDS = [
    "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://news.usni.org/feed",
    "https://gcaptain.com/feed/",
    "https://maritime-executive.com/articles.rss",
    "https://www.marinelink.com/news/rss",
    "https://feeds.feedburner.com/defense-update",
]

# Strategic keyword list — physical movements only (case-insensitive).
KEYWORDS = [
    "5th fleet", "b-52", "deployment", "aircraft carrier", "satellite imagery",
    "blockade", "strait", "naval drill", "force majeure", "choke point",
    "lng carrier", "supply chain",
]

# How far back a headline can be and still count as "live intelligence".
OSINT_LOOKBACK_HOURS = 48
MAX_ALERTS = 15


# ---------------------------------------------------------------------------
# 1. TECHNICAL ANALYSIS MODULE
# ---------------------------------------------------------------------------
def analyze_asset(name: str, ticker: str) -> dict:
    """Fetch daily data, compute MACD(12,26,9) + RSI(14), classify momentum."""
    result = {
        "name": name,
        "ticker": ticker,
        "ok": False,
        "signal": "N/A",
        "signal_ar": "غير متاح",
        "close": None,
        "rsi": None,
        "macd": None,
        "macd_signal": None,
        "error": None,
    }

    try:
        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        result["error"] = f"fetch error: {e}"
        return result

    if df is None or df.empty or len(df) < 40:
        result["error"] = "insufficient data returned"
        return result

    # yfinance may return a MultiIndex column set for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"]).copy()

    try:
        macd_line_s, signal_line_s = macd(df["Close"], fast=12, slow=26, signal=9)
        rsi_s = rsi(df["Close"], length=14)
    except Exception as e:
        result["error"] = f"indicator error: {e}"
        return result

    df = df.assign(_macd=macd_line_s, _signal=signal_line_s, _rsi=rsi_s)
    df = df.dropna(subset=["_macd", "_signal", "_rsi"])
    if df.empty:
        result["error"] = "no rows after indicator warm-up"
        return result

    last = df.iloc[-1]
    close = float(last["Close"])
    macd_line = float(last["_macd"])
    signal_line = float(last["_signal"])
    rsi_val = float(last["_rsi"])

    # --- Momentum rules ---
    if macd_line > signal_line and rsi_val < 65:
        signal, signal_ar = "BULLISH (Liquidity Inflow)", "صعودي — تدفّق سيولة داخلة"
    elif macd_line < signal_line and rsi_val > 35:
        signal, signal_ar = "BEARISH (Liquidity Outflow)", "هبوطي — تدفّق سيولة خارجة"
    else:
        signal, signal_ar = "NEUTRAL", "محايد"

    result.update(
        ok=True,
        signal=signal,
        signal_ar=signal_ar,
        close=round(close, 4),
        rsi=round(rsi_val, 2),
        macd=round(macd_line, 5),
        macd_signal=round(signal_line, 5),
    )
    return result


# ---------------------------------------------------------------------------
# 2. DEEP OSINT & LOGISTICS MODULE
# ---------------------------------------------------------------------------
def _entry_datetime(entry) -> dt.datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return dt.datetime(*t[:6])
            except Exception:
                pass
    return None


def scan_osint() -> list[dict]:
    """Scan feeds, keep only headlines matching strategic movement keywords."""
    cutoff = dt.datetime.utcnow() - dt.timedelta(hours=OSINT_LOOKBACK_HOURS)
    alerts = []
    seen_titles = set()

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
            haystack = f"{title} {summary}".lower()

            matched = [kw for kw in KEYWORDS if kw in haystack]
            if not matched:
                continue

            # Recency filter (skip only when a valid date exists and is stale)
            edt = _entry_datetime(entry)
            if edt is not None and edt < cutoff:
                continue

            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)

            alerts.append({
                "title": title,
                "link": getattr(entry, "link", ""),
                "source": source,
                "keywords": matched,
                "when": edt.strftime("%Y-%m-%d %H:%M") if edt else "—",
            })

    # Newest first where dates exist
    alerts.sort(key=lambda a: a["when"], reverse=True)
    return alerts[:MAX_ALERTS]


# ---------------------------------------------------------------------------
# 3. REPORTING & EMAIL MODULE (Arabic)
# ---------------------------------------------------------------------------
def build_report(tech: list[dict], alerts: list[dict]) -> str:
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("📊 التقرير الاستخباراتي للأسواق واللوجستيات")
    lines.append(f"التاريخ: {now}")
    lines.append("=" * 42)
    lines.append("")

    # --- Technical momentum ---
    lines.append("أولاً | الزخم الفني للسلع الاستراتيجية")
    lines.append("-" * 42)
    for t in tech:
        if t["ok"]:
            lines.append(f"• {t['name']}")
            lines.append(f"   الإشارة: {t['signal_ar']}")
            lines.append(f"   الإغلاق: {t['close']}  |  RSI: {t['rsi']}")
            lines.append(f"   MACD: {t['macd']}  ضد  الإشارة: {t['macd_signal']}")
        else:
            lines.append(f"• {t['name']}: تعذّر التحليل ({t['error']})")
        lines.append("")

    # --- OSINT alerts ---
    lines.append("ثانياً | تنبيهات الحركة اللوجستية والعسكرية (OSINT)")
    lines.append("-" * 42)
    if alerts:
        for a in alerts:
            tags = "، ".join(a["keywords"])
            lines.append(f"⚠️ {a['title']}")
            lines.append(f"   المصدر: {a['source']}  |  {a['when']}")
            lines.append(f"   الكلمات المفتاحية: {tags}")
            if a["link"]:
                lines.append(f"   الرابط: {a['link']}")
            lines.append("")
    else:
        lines.append("لا توجد تحركات ضمن الكلمات المفتاحية خلال نافذة الرصد.")
        lines.append("")

    # --- Analyst note: crude correlation hint (physical -> commodity) ---
    lines.append("ثالثاً | قراءة تحليلية موجزة")
    lines.append("-" * 42)
    note = synthesize_note(tech, alerts)
    lines.append(note)
    lines.append("")
    lines.append("— النظام اللاخادمي للاستخبارات | MVC Structure")
    return "\n".join(lines)


def synthesize_note(tech: list[dict], alerts: list[dict]) -> str:
    """Deterministic, non-fabricated linkage between OSINT load and momentum."""
    n = len(alerts)
    if n == 0:
        pressure = "خلفية هادئة لوجستياً"
    elif n <= 3:
        pressure = "ضغط لوجستي محدود"
    else:
        pressure = "ضغط لوجستي مرتفع على سلاسل الإمداد ونقاط الاختناق"

    parts = [f"مستوى الإشارات المادية: {pressure} ({n} تنبيه)."]
    for t in tech:
        if t["ok"]:
            parts.append(f"{t['name']} → {t['signal_ar']}.")
    parts.append(
        "ملاحظة: هذا الربط وصفي وليس تنبؤياً — التحركات المادية عند نقاط "
        "الاختناق قد تسبق تقلبات العرض، لكن الإشارة الفنية تبقى المرجع لتأكيد الاتجاه."
    )
    return " ".join(parts)


def send_email(subject: str, body: str) -> None:
    email = os.environ.get("EMAIL")
    app_password = os.environ.get("APP_PASSWORD")

    if not email or not app_password:
        print("[FATAL] EMAIL / APP_PASSWORD not set in environment.", file=sys.stderr)
        sys.exit(1)

    msg = MIMEMultipart()
    msg["From"] = email
    msg["To"] = email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(email, app_password)
            server.sendmail(email, email, msg.as_string())
        print("[OK] Report emailed successfully.")
    except Exception as e:
        print(f"[FATAL] Email send failed: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Serverless OSINT & Market Intelligence ===")

    tech = []
    for name, ticker in ASSETS.items():
        print(f"[..] Analyzing {name}")
        r = analyze_asset(name, ticker)
        status = r["signal"] if r["ok"] else f"FAILED: {r['error']}"
        print(f"     -> {status}")
        tech.append(r)

    print("[..] Scanning OSINT feeds")
    alerts = scan_osint()
    print(f"     -> {len(alerts)} strategic alert(s)")

    report = build_report(tech, alerts)
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    subject = f"تقرير الاستخبارات — النحاس والغاز | {today}"

    print("[..] Sending email")
    send_email(subject, report)
    print("=== Done ===")


if __name__ == "__main__":
    main()
