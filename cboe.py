import os
import requests
import datetime
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wallstreet as ws

if os.path.exists('credentials.txt'):
    with open('credentials.txt', 'r') as f:
        credentials = f.read().splitlines()
else:
    credentials = [None, None]

class CBOE:

    def __init__(self, CLIENT_ID=credentials[0], CLIENT_SECRET=credentials[1]):
        if not CLIENT_ID and not CLIENT_SECRET:
            raise ValueError('Could not read credentials from local credentials.txt. Please input valid credentials as arguments.')

        self.client_id = CLIENT_ID
        self.client_secret = CLIENT_SECRET
        self.connect()
        self.today = datetime.datetime.today()


    def connect(self):
        identity_url = "https://id.livevol.com/connect/token"
        authorization_token  = base64.b64encode((self.client_id + ':' + self.client_secret).encode())
        headers = {"Authorization": "Basic " + authorization_token.decode('ascii')}
        payload = {"grant_type": "client_credentials"}

        token_data = requests.post(identity_url, data=payload, headers=headers)

        if token_data.status_code == 200:
            self.access_token = token_data.json()['access_token']
            if not len(self.access_token) > 0:
                print('Authentication failed')
        else:
            print("Authentication failed")        

    def convert_exp_shorthand(self,date:str):
        month_map = {
            'jan':1,
            'feb':2,
            'mar':3,
            'apr':4,
            'may':5,
            'jun':6,
            'jul':7,
            'aug':8,
            'sep':9,
            'oct':10,
            'nov':11,
            'dec':12
        }
        if date[0] == 'e':

            month = month_map[date[1:4].lower()]
            year = int(date[4:])
            if len(str(year)) == 2:
                year = 2000 + year
            return self.get_opex_date(month,year)
        else:
            raise ValueError('Invalid date format - must be in the format {month}e{year}')


    @staticmethod
    def get_opex_date(month,year):
        d = datetime.date(year,month,1)
        d += datetime.timedelta( (4-d.weekday()) % 7 )
        d += datetime.timedelta(14)
        return d

    @staticmethod
    def dealer_pos(option_type):
        if option_type == 'C':
            return 1
        elif option_type == 'P':
            return -1

    def get_quote(self, ticker, option_type='C'):
        today = self.today.strftime('%Y-%m-%d')
        max_exp = (self.today + datetime.timedelta(days=365)).strftime('%Y-%m-%d')

        self.stock = ws.Stock(ticker)
        spot = self.stock.price
        min_k = int(spot * 0.7)
        max_k = int(spot * 1.3)

        url = f'https://api.livevol.com/v1/live/allaccess/market/option-and-underlying-quotes?root={ticker}&option_type={option_type}&date={today}&min_expiry={today}&max_expiry={max_exp}&min_strike={min_k}&max_strike={max_k}&symbol={ticker}'
        headers = {"Authorization": "Bearer " + self.access_token}
        data = requests.get(url, headers=headers)
        
        return data
        
    def get_options(self, ticker, option_type='C'):
        r = self.get_quote(ticker,option_type)
        df = pd.DataFrame(r.json().get('options'))
        df['expiry'] = pd.to_datetime(df.expiry)
        df['dealer_pos'] = df.option_type.apply(self.dealer_pos)
        df = df.assign(
            exp_month = df.expiry.dt.month,
            exp_year = df.expiry.dt.year,
            exp_day = df.expiry.dt.day,
            agg_gamma = df.gamma * df.open_interest,
            dealer_gamma = df.gamma * df.open_interest * df.dealer_pos * 100 * self.stock.price
        )

        return df

    