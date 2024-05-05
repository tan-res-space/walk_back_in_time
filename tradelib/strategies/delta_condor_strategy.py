from ._strategy import TradeStrategy
from trading_platform import TradingPlatform
from tradelib.strategies.strategy_components.delta_condor_component import DeltaCondorComponent
from models.Portfolio import Portfolio
from models.Blotter import Blotter
from models.Backtest import Backtest
from tradelib.strategies.strategy_components.hedge_component import HedgeComponent
from tradelib.strategies.strategy_components.unwind_component import UnwindComponent

from tradelib_global_constants import underlying, delta_otm_tolerance, strict_condor, strict_tolerance, trade_interval_time, unit_size, unwind_time, delta_outstrike, hedge_interval_time

class DeltaCondorStrategy(TradeStrategy):
    def __init__(self, trading_platform: TradingPlatform, portfolio: Portfolio, blotter: Blotter) -> None:
        super().__init__("delta_condor_strategy", trading_platform, portfolio, blotter)

        unwind_component = UnwindComponent(trading_platform = trading_platform, 
                                            portfolio = self.portfolio, 
                                            blotter = self.blotter)
        self.add_component(unwind_component)

        delta_condor_component = DeltaCondorComponent(trading_platform = trading_platform, 
                                                        portfolio = self.portfolio, 
                                                        blotter = self.blotter, 
                                                        skip_count=trade_interval_time, 
                                                        delta_outstrike=delta_outstrike, 
                                                        tolerance=delta_otm_tolerance, 
                                                        strict_condor=strict_condor, 
                                                        strict_tolerance=strict_tolerance, 
                                                        underlying=underlying, 
                                                        unit_size=unit_size, 
                                                        unwind_time=unwind_time)
        self.add_component(delta_condor_component)

        hedge_component = HedgeComponent(trading_platform = trading_platform, 
                                            portfolio = self.portfolio, 
                                            blotter = self.blotter, 
                                            skip_count = hedge_interval_time)

        self.add_component(hedge_component)
