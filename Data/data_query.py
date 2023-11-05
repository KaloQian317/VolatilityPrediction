import yfinance as yf
import pandas as pd
import numpy as np

path = 'IWB_holdings.csv'

data = pd.read_csv(path,skiprows = 9)
tickernames = data['Ticker'].values[:-2]
tickernames = np.delete(tickernames,-3)

num_of_ticker = 0

# Loop through the tickers and fetch intraday data
for ticker in tickernames:
    # Download intraday data at a 5-minute interval within last 60 days
    data = yf.download(ticker, period='60d', interval="5m")
    # Add the ticker name to each downloaded data
    data['Ticker'] = ticker

    if num_of_ticker == 0:
        intraday_data = data.copy()
    else:
    # Concatenate data with the intraday_data DataFrame
        intraday_data = pd.concat([intraday_data, data])
    num_of_ticker += 1

# Save the data to a CSV file for visualization
intraday_data.to_csv('russell_1000_intraday_data.csv')
# Save the data to a pkl file for python to read
intraday_data.to_pickle('russell_1000_intraday_data.pkl')

