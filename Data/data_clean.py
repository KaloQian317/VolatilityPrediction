import pandas as pd
import numpy as np
<<<<<<< HEAD
=======
import datetime as dt
>>>>>>> 84536fcabe7fc0fab9f79dbd1bc510f49dffa6f8

#################################### Data Adjustment ########################################################
def drop_ticker (intraday_data, threshold_num=None):
    '''
    intraday_data: pandas dataframe of intraday data
    threshold_num: int number that requires the function to drop tickers with number of data less than the threshold
    return: pandas dataframe of cleaned intraday data
    '''
<<<<<<< HEAD
    data_length = intraday_data.groupby('Ticker').count().Open
=======
    data_length = len(intraday_data)
    # BUG: np.quantile only accept array-like real numbers
    """
>>>>>>> 84536fcabe7fc0fab9f79dbd1bc510f49dffa6f8
    if threshold_num is None:
        q1 = np.quantile(data_length, 0.25)
        q3 = np.quantile(data_length, 0.75)
        iqr = q3-q1
        threshold_num = q1-(1.5*iqr)
<<<<<<< HEAD
    tmp = data_length >= threshold_num
=======
    """
    tmp = intraday_data.groupby("Ticker").count().Open >= threshold_num
>>>>>>> 84536fcabe7fc0fab9f79dbd1bc510f49dffa6f8
    ticker_name = tmp.loc[tmp==True].index
    cleaned_intraday_data = intraday_data.reset_index().set_index('Ticker').loc[ticker_name]
    cleaned_intraday_data.rename(columns={'index':'timestamp'}, inplace=True)

    return cleaned_intraday_data.reset_index().set_index('timestamp')

def reconstruct_dataframe(cleaned_data):
    # Need to rewrite the method. Too long to run.
    '''
    cleaned_data: pandas dataframe of cleaned intraday data
    return: pandas dataframe of reconstructed intraday data, missing value is set to None
    '''
    # Use a ticker with full timestamps to build a empty sample
    data_length = cleaned_data.groupby('Ticker').count().Open
    full_ticker = data_length.index[np.argmax(data_length)]
    sample = cleaned_data.loc[cleaned_data['Ticker']==full_ticker].copy()
    full_timestamps = sample.index.to_numpy()
    sample.iloc[:,0:] = np.nan
    
    # Groupby tickers
    g1 = cleaned_data.groupby('Ticker',group_keys = False)
    
    # For each ticker, put value in the 
    reconstructed_data = pd.DataFrame()

    for ticker,df in g1:
        temp = sample.copy()
        common_indices = df.index.intersection(temp.index)
        temp.loc[common_indices] = df.loc[common_indices]
        temp.loc[:,'Ticker'] = ticker
        reconstructed_data = pd.concat([reconstructed_data,temp])
    
    return reconstructed_data

def impute_missing_data (reconstructed_data):
    '''
    reconstructed_data: pandas dataframe of cleaned intraday data after reconstructed with missing None value
    return: pandas dataframe of imputed intraday data, missing value is linearly interpolated
    '''
    # Impute missing data for the reconstructed tickers.
    def ticker_interpolate(df):
        if df.iloc[:,1].isnull().any():
            df.iloc[:,1:] = df.iloc[:,1:].interpolate(method = 'linear')
        return df
    
    g2 = reconstructed_data.groupby('Ticker',group_keys = False)
    imputed_data = g2.apply(ticker_interpolate)
    # There might be some data left without imputed at the start and the end of the day
    return imputed_data

"""
# Clean the data and save in local
intraday_data = pd.read_pickle('russell_1000_intraday_data.pkl')
<<<<<<< HEAD
cleaned_intraday_data = drop_ticker (intraday_data)
=======
threshold_num = 4600
cleaned_intraday_data = drop_ticker (intraday_data, threshold_num)
reconstructed_intraday_data = reconstruct_dataframe(cleaned_intraday_data)
reconstructed_intraday_data.to_pickle("reconstructed_intraday_data.pkl")
imputed_intraday_data = impute_missing_data(reconstructed_intraday_data)
imputed_intraday_data.to_pickle("imputed_intraday_data.pkl")

>>>>>>> 84536fcabe7fc0fab9f79dbd1bc510f49dffa6f8
print()
"""

#################################### Volatility ########################################################
def cal_realized_vol(cleaned_intraday_data):
    # Define a function to calculate daily vol
    def cal_intraday_R(table):
        # Here, table is a pandas dataframe of cleaned intraday data but only for one ticker.
        # This function gives intraday return for each 5-minute interval
        tmp = table.copy()
        tmp['Intraday Return'] = (tmp['Adj Close'] - tmp['Open'])/tmp['Open']
        return tmp
    # calculate intraday return for each 5-minute internal
    intraday_data = cleaned_intraday_data.groupby('Ticker').apply(cal_intraday_R)
    intraday_data['R2'] = intraday_data['Intraday Return'] ** 2
    intraday_data = intraday_data.drop(columns='Ticker').reset_index('Ticker')

    def cal_daily_rv(table):
        # calculate daily rv and drop first and last 30 minutes for every day
        tmp = table.copy()
        def get_oneday_rvf1ticker(table):
            #drop first and last 30 minutes for every day and calculate daily rv on that date for a ticker
            table = table.loc[(table.index.time >= dt.time(10, 00)) & (table.index.time <= dt.time(15, 25))]
            table['rv'] = (table['R2'].sum()) ** (1/2)
            return table[['Ticker', 'rv']].iloc[0]
        tmp = tmp.groupby(tmp.index.date).apply(get_oneday_rvf1ticker)
        return tmp

    rv = intraday_data.groupby('Ticker').apply(cal_daily_rv)
    return rv

imputed_intraday_data = pd.read_pickle("imputed_intraday_data.pkl")
rv = cal_realized_vol(imputed_intraday_data)
print(rv)