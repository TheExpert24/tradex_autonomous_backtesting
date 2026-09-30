import numpy as np

ENTRY_THRESHOLD = 0.12


def signal(df):
    if len(df) < 40:
        return 0.0

    close = df["close"]
    volume = df["volume"]

    returns = close.pct_change()
    vol = returns.std()

    if np.isnan(vol):
        return 0.0

    vwap = (
        (close * volume).cumsum()
        / (volume.cumsum() + 1e-8)
    )

    dev = (
        (close - vwap)
        / (vwap + 1e-8)
    )

    z = (
        (dev - dev.mean())
        / (dev.std() + 1e-8)
    )

    vwap_signal = -z.iloc[-1]

    mom10 = (
        close.iloc[-1]
        / close.iloc[-10]
        - 1
    )

    mom20 = (
        close.iloc[-1]
        / close.iloc[-20]
        - 1
    )

    momentum = (
        0.6 * mom10
        + 0.4 * mom20
    )

    trend_slope = np.polyfit(
        np.arange(30),
        close.iloc[-30:],
        1
    )[0]

    trend_strength = (
        abs(trend_slope)
        / close.iloc[-1]
    )

    trend_regime = (
        abs(momentum)
        + trend_strength
    )

    trending = trend_regime > 0.01

    if trending:
        alpha = (
            0.60 * momentum
            + 0.30 * trend_slope
            + 0.10 * vwap_signal
        )
    else:
        alpha = (
            0.60 * vwap_signal
            + 0.30 * momentum
            + 0.10 * trend_slope
        )

    alpha *= (
        1 - min(vol * 40, 1.0)
    )

    if abs(alpha) < ENTRY_THRESHOLD:
        return 0.0

    return float(alpha)
