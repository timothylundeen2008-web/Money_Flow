"""
market_time.py  (v1 — August 2026)
──────────────────────────────────
Eastern-time display and trading-calendar helpers, shared by all three
dashboards and the scheduled logger.

WHY THIS EXISTS
───────────────
Every timestamp in the stack was rendered with a naive `datetime.now()`,
which returns the SERVER's local time. On Streamlit Community Cloud that is
UTC — so a dashboard refreshed at 4:05pm ET displayed "20:05", and a user
reading it as Eastern would believe the data was four to five hours staler
(or fresher) than it actually was. For a system whose entire staleness
discipline depends on knowing when a number was captured, an unlabeled
wrong-zone timestamp is worse than no timestamp.

Everything here is timezone-AWARE. The rule adopted across the codebase:

    * DISPLAY in America/New_York, always with an explicit ET/EDT/EST label
    * STORE in UTC ISO-8601, always
    * KEY logs by the ET trading date, never by UTC date

That last point matters more than it looks. A job running at 22:30 UTC in
summer is 18:30 ET the SAME day, but at 01:30 UTC in winter it would be
20:30 ET the PREVIOUS day. Keying a daily log by UTC date would silently
mislabel every winter entry.

DST IS HANDLED BY zoneinfo, NOT BY ARITHMETIC
─────────────────────────────────────────────
No fixed -5/-4 offsets anywhere. ZoneInfo("America/New_York") resolves the
correct offset for any given instant, including the transition weekends.
"""

from __future__ import annotations

from datetime import datetime, date, time, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# US equity market holidays. Extend annually — an unmaintained list silently
# starts logging holidays as missed sessions, so `holidays_through` below
# reports the last covered year and the caller can warn.
MARKET_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
HOLIDAYS_THROUGH = 2027

# Early closes (1:00pm ET). Data settles earlier on these days.
EARLY_CLOSES = {
    "2026-11-27", "2026-12-24",
    "2027-11-26",
}


def now_et() -> datetime:
    """Current time, timezone-aware, in Eastern."""
    return datetime.now(ET)


def now_utc() -> datetime:
    return datetime.now(UTC)


def to_et(dt: datetime) -> datetime:
    """Convert any datetime to Eastern. Naive input is ASSUMED UTC.

    That assumption is deliberate: naive datetimes in this codebase come from
    servers running UTC. Assuming local time would produce a plausible-looking
    but wrong timestamp, which is the failure mode being fixed.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ET)


def fmt_et(dt: datetime | None = None, fmt: str = "%b %d, %Y %I:%M %p") -> str:
    """Display string with an explicit zone label — e.g. 'Aug 09, 2026 07:27 AM EDT'.

    The label is never omitted. An unlabeled timestamp is what created the
    original ambiguity.
    """
    d = to_et(dt) if dt else now_et()
    return f"{d.strftime(fmt)} {d.tzname()}"


def fmt_et_short(dt: datetime | None = None) -> str:
    d = to_et(dt) if dt else now_et()
    return f"{d.strftime('%I:%M %p')} {d.tzname()}"


def et_date(dt: datetime | None = None) -> date:
    """The ET calendar date — the correct key for any daily log."""
    return (to_et(dt) if dt else now_et()).date()


def is_holiday(d: date | None = None) -> bool:
    return (d or et_date()).isoformat() in MARKET_HOLIDAYS


def is_trading_day(d: date | None = None) -> bool:
    d = d or et_date()
    return d.weekday() < 5 and not is_holiday(d)


def calendar_stale() -> bool:
    """True when the holiday list no longer covers the current year."""
    return et_date().year > HOLIDAYS_THROUGH


def close_time(d: date | None = None) -> time:
    d = d or et_date()
    return time(13, 0) if d.isoformat() in EARLY_CLOSES else MARKET_CLOSE


def is_market_open(dt: datetime | None = None) -> bool:
    d = to_et(dt) if dt else now_et()
    if not is_trading_day(d.date()):
        return False
    return MARKET_OPEN <= d.time() < close_time(d.date())


def minutes_since_close(dt: datetime | None = None) -> float | None:
    """Minutes since today's close. None when today isn't a trading day."""
    d = to_et(dt) if dt else now_et()
    if not is_trading_day(d.date()):
        return None
    close_dt = datetime.combine(d.date(), close_time(d.date()), tzinfo=ET)
    return (d - close_dt).total_seconds() / 60.0


def last_trading_day(d: date | None = None) -> date:
    """Most recent trading day at or before `d`."""
    d = d or et_date()
    for _ in range(10):
        if is_trading_day(d):
            return d
        d -= timedelta(days=1)
    return d


def market_status() -> dict:
    """Everything a dashboard header needs, in one call."""
    n = now_et()
    trading = is_trading_day(n.date())
    open_now = is_market_open(n)
    mins = minutes_since_close(n)

    if not trading:
        label = "Holiday" if is_holiday(n.date()) else "Weekend"
        status = f"Market closed — {label}"
    elif open_now:
        status = "Market open"
    elif n.time() < MARKET_OPEN:
        status = "Pre-market"
    else:
        status = "After hours"

    return {
        "now_et": n,
        "display": fmt_et(n),
        "is_trading_day": trading,
        "is_open": open_now,
        "status": status,
        "minutes_since_close": mins,
        "et_date": n.date().isoformat(),
        "utc_iso": now_utc().isoformat(timespec="seconds"),
        "calendar_stale": calendar_stale(),
    }


def selftest() -> dict:
    """Verify DST handling and calendar logic on known dates."""
    failures = []

    # DST transition weekends — offsets must differ.
    summer = datetime(2026, 7, 15, 12, 0, tzinfo=UTC).astimezone(ET)
    winter = datetime(2026, 1, 15, 12, 0, tzinfo=UTC).astimezone(ET)
    if summer.utcoffset() == winter.utcoffset():
        failures.append("DST not applied — summer and winter offsets identical")
    if summer.tzname() != "EDT":
        failures.append(f"July should be EDT, got {summer.tzname()}")
    if winter.tzname() != "EST":
        failures.append(f"January should be EST, got {winter.tzname()}")

    # The winter-UTC-date trap: 01:30 UTC on Jan 6 is Jan 5 in ET.
    late = datetime(2026, 1, 6, 1, 30, tzinfo=UTC)
    if et_date(late) != date(2026, 1, 5):
        failures.append(f"UTC->ET date rollover wrong: {et_date(late)}")

    if is_trading_day(date(2026, 7, 4)) or is_trading_day(date(2026, 7, 5)):
        failures.append("Weekend/holiday flagged as trading day")
    if not is_trading_day(date(2026, 8, 7)):
        failures.append("2026-08-07 (Friday) should be a trading day")
    if is_trading_day(date(2026, 12, 25)):
        failures.append("Christmas flagged as a trading day")
    if close_time(date(2026, 11, 27)) != time(13, 0):
        failures.append("Early close not detected")

    return {"ok": not failures, "failures": failures,
            "now": fmt_et(), "status": market_status()["status"],
            "calendar_stale": calendar_stale()}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, default=str))
