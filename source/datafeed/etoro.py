import json
import pandas as pd
import requests

class SymbolScraper:
    
    
    def __init__(self):
        # Define the links for the scraping
        instruments_link = 'https://api.etorostatic.com/sapi/app-data/web-client/app-data/instruments-groups.json'
        data_link = 'https://api.etorostatic.com/sapi/instrumentsmetadata/V1.1/instruments/bulk?bulkNumber=1&totalBulks=1'
        
        # Gather types of instruments and their attributes
        response = requests.get(instruments_link)
        parsed_types = json.loads(response.text)
        
        # Divide types
        self.instruments = parsed_types['InstrumentTypes']
        self.exchanges = parsed_types['ExchangeInfo']
        self.stocks = parsed_types['StocksIndustries']
        self.crypto = parsed_types['CryptoCategories']
        
        # Gather all the instruments
        response = requests.get(data_link)
        self.data = json.loads(response.text)['InstrumentDisplayDatas']
        
        # We collect the instruments with their attributes here
        self.inst = []

        
    def replace_symbol_ending(self,symbol,old_end,new_end):
        if symbol.endswith(old_end):
                i = symbol.rsplit(old_end,1)
                symbol = new_end.join(i)

        return symbol
    def get(self):
        # Loop through all the instruments
        symbols = []

        for d in self.data:
        
            # NEW EDIT: If the instrument is not available for the users, we don't need it
            if d['IsInternalInstrument']:
                continue
        
            # Gather the necessary data about the instrument
            instrument_typeID = d['InstrumentTypeID']
            name = d['InstrumentDisplayName']
            exchangeID = d['ExchangeID']
            symbol = d['SymbolFull']

            symbol = self.replace_symbol_ending(symbol,'.NV','.AS')
            symbol = self.replace_symbol_ending(symbol,'.ZU','.SW')
            symbol = self.replace_symbol_ending(symbol,'.B','-B')
            symbol = symbol.replace("00241.HK", "0241.HK")
            symbol = symbol.replace("02018.HK", "2018.HK")
            symbol = symbol.replace("02020.HK", "2020.HK")
            symbol = symbol.replace("00285.HK", "0285.HK")
            symbol = symbol.replace("01211.HK", "1211.HK")
            symbol = symbol.replace("03690.HK", "3690.HK")
            symbol = symbol.replace("BOSSD.DE", "BOSS.DE")
            symbol = symbol.replace("LSXD.DE", "LXS.DE")



            # Instrument type
            instrument_type = next(item for item in self.instruments
                                   if item['InstrumentTypeID'] == instrument_typeID)['InstrumentTypeDescription']
        
            # Industry type
            try:
                industryID = d['StocksIndustryID']
                industry = next(item for item in self.stocks
                                if item['IndustryID'] == industryID)['IndustryName']
            # If the instrument don't have industry, we have to give it a placeholder
            except (KeyError, StopIteration):
                industry = '-'
        
            # Exchange location
            try:
                exchange = next(item for item in self.exchanges
                                if item['ExchangeID'] == exchangeID)['ExchangeDescription']
            # If the instrument don't have exchange location, we have to give it a placeholder
            except StopIteration:
                exchange = '-'
        
            # Sum up the gathered data
            self.inst.append({
                'name': name,
                'symbol': symbol,
                'instrument type': instrument_type,
                'exchange': exchange,
                'industry': industry
            })

            symbols.append(symbol)
        
        
        # Create a Pandas DataFrame from the assets
        self.inst = pd.DataFrame(self.inst, index=symbols)
        return self.inst