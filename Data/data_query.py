import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import pytz

def query_data (path, save_as_csv=True, save_as_pkl=True):
    """
    Query the newest 5-min intraday data for 60 days from Yahoo Finance.

    save_as_csv: whether to save the data as .csv file in local.
    save_as_pkl: whether to save the data as .pkl file in local.

    return: a pandas dataframe of uncleaned 5-min intraday data.
    """
#     path = 'IWB_holdings.csv'

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
    if save_as_csv == True:
        intraday_data.to_csv('russell_1000_intraday_data.csv')
    # Save the data to a pkl file for python to read
    if save_as_pkl == True:
        intraday_data.to_pickle('russell_1000_intraday_data.pkl')

    return intraday_data

def update_data (path,use_original_ticker=True, save_as_csv=True, save_as_pkl=True):
    """
    Query the newest 5-min intraday data starting from the latest date in the old dataframe, and append the queried data
    onto the old dataframe.

    use_original_ticker: whether to use the ticker name from "IWB_holdings.csv"
    save_as_csv: whether to save the data as .csv file in local.
    save_as_pkl: whether to save the data as .pkl file in local.

    return: a pandas dataframe of updated and uncleaned 5-min intraday data.
    """
    intraday_data = pd.read_pickle(path)
    latest_date = (intraday_data.index.max().to_pydatetime().date() + dt.timedelta(days=1)).strftime('%Y-%m-%d') # Add one more day
    new_york_tz = pytz.timezone('America/New_York')
    current_date = datetime.now(new_york_tz).date().strftime('%Y-%m-%d')

    # If use original ticker names, use the ticker name from IWB_holdings.
    if use_original_ticker == True:
        data = pd.read_csv('IWB_holdings.csv', skiprows=9)
        tickernames = data['Ticker'].values[:-2]
        tickernames = np.delete(tickernames, -3)
    # If don't use original ticker names, use the ticker name from the latest table.
    else:
        tickernames = intraday_data.Ticker.drop_duplicates().to_numpy()


    num_of_ticker = 0
    # Loop through the tickers and fetch intraday data
    for ticker in tickernames:
        # Download intraday data at a 5-minute interval within last 60 days
        data = yf.download(ticker, start=latest_date, end=current_date, interval="5m")
        # Add the ticker name to each downloaded data
        data['Ticker'] = ticker
        if num_of_ticker == 0:
            updated_intraday_data = data.copy()
        else:
            # Concatenate data with the intraday_data DataFrame
            updated_intraday_data = pd.concat([updated_intraday_data, data])
        num_of_ticker += 1

    # Concatenate two tables (Need to test this function)
    intraday_data = pd.concat([intraday_data, updated_intraday_data])

    # Save the data to a CSV file for visualization
    if save_as_csv == True:
        intraday_data.to_csv('russell_1000_intraday_data.csv')
    # Save the data to a pkl file for python to read
    if save_as_pkl == True:
        intraday_data.to_pickle('russell_1000_intraday_data.pkl')

    return intraday_data

# update_data()