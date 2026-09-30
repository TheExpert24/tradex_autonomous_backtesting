from AlgorithmImports import *

from collections import deque
from datetime import timedelta
import numpy as np
import pandas as pd

from vwap_strategy import signal


class VwapRankedTrader(QCAlgorithm):

    START_DATE = (2024, 1, 1)
    END_DATE = (2025, 1, 1)
    INITIAL_CASH = 100000

    # Your exact universe
    SYMBOLS = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
        "AMD", "AVGO", "NFLX", "TSLA", "PLTR", "MU", "CRWD",
        "NOW", "UBER", "SHOP", "PANW", "SNOW", "ARM", "SMCI",
        "JPM", "GS", "MS", "BAC", "V", "MA", "XOM", "CVX", "LLY",
        "UNH", "WMT", "COST", "HD", "MCD", "QCOM", "ORCL", "IBM",
        "ADBE", "CRM", "SPY", "QQQ", "IWM"
    ]

    BAR_MINUTES = 5
    LOOKBACK_BARS = 60

    MAX_POSITIONS = 25
    RISK_PER_TRADE = 0.001

    ENTRY_Z = 0.75
    EXIT_Z = 0.0

    STOP_LOSS_PCT = 0.03
    COOLDOWN_MINUTES = 5

    def initialize(self):
        self.set_start_date(*self.START_DATE)
        self.set_end_date(*self.END_DATE)
        self.set_cash(self.INITIAL_CASH)

        self.symbols = []

        for ticker in self.SYMBOLS:
            security = self.add_equity(
                ticker,
                Resolution.MINUTE
            )

            security.set_data_normalization_mode(
                DataNormalizationMode.Raw
            )

            self.symbols.append(security.symbol)

        self.windows = {
            symbol: deque(maxlen=self.LOOKBACK_BARS)
            for symbol in self.symbols
        }

        self.consolidators = {}
        self.last_trade_time = {}

        self.settings.free_portfolio_value_percentage = 0.025

        self.set_warm_up(
            self.BAR_MINUTES * (self.LOOKBACK_BARS + 10),
            Resolution.MINUTE
        )

        for symbol in self.symbols:
            consolidator = TradeBarConsolidator(
                timedelta(minutes=self.BAR_MINUTES)
            )

            consolidator.data_consolidated += (
                self.on_consolidated
            )

            self.subscription_manager.add_consolidator(
                symbol,
                consolidator
            )

            self.consolidators[symbol] = consolidator

        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(
                timedelta(minutes=self.BAR_MINUTES)
            ),
            self.rebalance
        )

    def on_data(self, data):
        pass

    def on_consolidated(self, sender, bar):
        symbol = bar.symbol

        if symbol not in self.windows:
            return

        self.windows[symbol].append({
            "time": bar.end_time,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume)
        })

    def dataframe_for(self, symbol):
        rows = list(self.windows[symbol])

        if len(rows) < 40:
            return None

        return pd.DataFrame(rows).set_index("time")

    def rank_signals(self):
        ranked = []

        for symbol in self.symbols:
            df = self.dataframe_for(symbol)

            if df is None:
                continue

            try:
                alpha = float(signal(df))
            except Exception as e:
                self.debug(
                    f"Signal error {symbol.Value}: {e}"
                )
                continue

            if not np.isfinite(alpha):
                continue

            ranked.append({
                "symbol": symbol,
                "alpha": alpha,
                "price": float(
                    df["close"].iloc[-1]
                )
            })

        if not ranked:
            return []

        alphas = np.array(
            [x["alpha"] for x in ranked],
            dtype=float
        )

        mean = float(np.mean(alphas))
        std = float(np.std(alphas))

        if std < 1e-8:
            std = 1.0

        for stock in ranked:
            stock["z"] = (
                (stock["alpha"] - mean) / std
            )

        ranked.sort(
            key=lambda x: x["z"],
            reverse=True
        )

        return ranked

    def can_trade(self, symbol):
        last = self.last_trade_time.get(symbol)

        if last is None:
            return True

        return (
            self.time - last
            >= timedelta(
                minutes=self.COOLDOWN_MINUTES
            )
        )

    def allocation_for(self, equity, z):
        strength = min(abs(z), 3.0)

        return (
            equity
            * self.RISK_PER_TRADE
            * strength
        )

    def apply_stop_losses(self):
        for symbol in self.symbols:
            holding = self.portfolio[symbol]

            if not holding.invested:
                continue

            if (
                holding.unrealized_profit_percent
                <= -self.STOP_LOSS_PCT
            ):
                self.debug(
                    f"STOP LOSS {symbol.Value} "
                    f"{holding.unrealized_profit_percent:.2%}"
                )

                self.liquidate(
                    symbol,
                    tag="STOP_LOSS"
                )

                self.last_trade_time[symbol] = self.time

    def execute_exits(self, ranked):
        for stock in ranked:
            symbol = stock["symbol"]

            if (
                stock["z"] < self.EXIT_Z
                and self.portfolio[symbol].invested
                and self.can_trade(symbol)
            ):
                self.liquidate(
                    symbol,
                    tag=f"SELL z={stock['z']:.2f}"
                )

                self.last_trade_time[symbol] = self.time

    def execute_entries(self, ranked, equity):

        open_positions = sum(
            1
            for symbol in self.symbols
            if self.portfolio[symbol].invested
        )

        for stock in ranked:

            if open_positions >= self.MAX_POSITIONS:
                break

            symbol = stock["symbol"]

            if stock["z"] <= self.ENTRY_Z:
                continue

            if self.portfolio[symbol].invested:
                continue

            if not self.can_trade(symbol):
                continue

            allocation = self.allocation_for(
                equity,
                stock["z"]
            )

            if allocation <= 0:
                continue

            target_weight = allocation / equity

            if target_weight > 0.03:
                target_weight = 0.03

            self.set_holdings(
                symbol,
                target_weight,
                tag=f"BUY z={stock['z']:.2f}"
            )

            self.last_trade_time[symbol] = self.time
            open_positions += 1

    def rebalance(self):

        if self.is_warming_up:
            return

        if not self.securities[
            self.symbols[0]
        ].exchange.hours.is_open(
            self.time,
            extended_market_hours=False
        ):
            return

        self.apply_stop_losses()

        ranked = self.rank_signals()

        if not ranked:
            return

        equity = float(
            self.portfolio.total_portfolio_value
        )

        if equity <= 0:
            return

        # Exits first
        self.execute_exits(ranked)

        # Then entries
        self.execute_entries(
            ranked,
            equity
        )

        open_positions = sum(
            1
            for symbol in self.symbols
            if self.portfolio[symbol].invested
        )

        self.debug(
            f"{self.time} | "
            f"Signals={len(ranked)} | "
            f"Positions={open_positions} | "
            f"Equity=${equity:,.2f}"
        )

    def on_end_of_algorithm(self):
        self.debug(
            f"Final equity: "
            f"${self.portfolio.total_portfolio_value:,.2f}"
        )
