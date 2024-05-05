'''
this class is responsible for building strategies
'''
from ._strategy import TradeStrategy
from .static_condor_strategy import StaticCondorStrategy
from .delta_condor_strategy import DeltaCondorStrategy
from .constant_risk_static_condor_strategy import ConstantRiskStaticCondorStrategy
from trading_platform import TradingPlatform
from models.Portfolio import Portfolio
from models.Blotter import Blotter
from models.Backtest import Backtest
from .general_strategy import GeneralStrategy
from .strategy_components._strategy_component import StrategyComponent
from typing import List

class StrategyFactory():
    def build(self, strategy_name: str, component_list: List[StrategyComponent], trading_platform: TradingPlatform, portfolio: Portfolio, blotter: Blotter, backtest: Backtest):
        if (len(component_list) == 0):
            return self.fromName(strategy_name, trading_platform, portfolio, blotter)
        else:
            return self.fromComponentList(strategy_name, component_list, trading_platform, portfolio, blotter, backtest)


    def fromName(self, strategy_name: str, trading_platform: TradingPlatform, portfolio: Portfolio, blotter: Blotter) -> TradeStrategy:
        if strategy_name == "STATIC_CONDOR_STRATEGY":
            return StaticCondorStrategy(trading_platform, portfolio, blotter)

        elif strategy_name == "DELTA_CONDOR_STRATEGY":
            return DeltaCondorStrategy(trading_platform, portfolio, blotter)

        elif strategy_name == "CONSTANT_RISK_CONDOR_STRATEGY":
            return ConstantRiskStaticCondorStrategy(trading_platform, portfolio, blotter)
        
    def fromComponentList(self, strategy_name: str, component_list: List[StrategyComponent], trading_platform: TradingPlatform, portfolio: Portfolio, blotter: Blotter):
        return GeneralStrategy(strategy_name, component_list, trading_platform, portfolio, blotter)
