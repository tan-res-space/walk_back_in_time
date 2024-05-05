from models.BlotterPacket import BlotterPacket
from typing import List, Dict
from models.Trade import Trade
from datetime import datetime
from tradelib_utils import get_blotter_filename, serialiseInstrument
import os
import csv
from tradelib_logger import logger

_logger = logger.getLogger("blotter")
class Blotter:
    def __init__(self, output_dir:str):
        self.blotter_list: List[BlotterPacket] = []
        self.packet_id = 1
        self.trade_id = 1
        self.blotter_dir = os.path.join(output_dir, 'blotter')
        os.makedirs(self.blotter_dir, exist_ok=True)
        self.component_trade_id: Dict[str, int] = {}

    def addTradeList(self, trade_list:List[Trade], trade_component: str, timestamp: datetime):
        for trade in trade_list:
            if trade_component not in self.component_trade_id:
                self.component_trade_id[trade_component] = 1
            self.blotter_list.append(BlotterPacket(trade, self.trade_id, self.packet_id, self.component_trade_id[trade_component], trade_component, timestamp))
            self.trade_id += 1
        self.packet_id += 1
        self.component_trade_id[trade_component] += 1

    def saveToCsvAndClear(self, start_timestamp: datetime, end_timestamp: datetime):
        '''
        saves current state of portfolio in a csv
        '''
        file_path = os.path.join(self.blotter_dir, get_blotter_filename(start_timestamp, end_timestamp))
        with open(file_path, "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(['timestamp', 'trade_id', 'packet_id', 'component_trade_id', 'trade_component', 'instrument_id', 'instrument_id_key ', 'position', 'trade_sub_type','instrument_object'])
            for blotter_packet in self.blotter_list:
                writer.writerow([blotter_packet.timestamp, blotter_packet.trade_id, blotter_packet.packet_id, blotter_packet.component_trade_id, blotter_packet.trade_component, blotter_packet.trade.instrument.id, blotter_packet.trade.instrument.idKey(), blotter_packet.trade.position, blotter_packet.trade.trade_sub_type, serialiseInstrument(blotter_packet.trade.instrument)])
        self.trade_id = 1
        self.packet_id = 1
        self.component_trade_id.clear()
        self.blotter_list.clear()