import pandas as pd
import numpy as np
import datetime as dt

#################################### Data Adjustment ########################################################
def drop_ticker (intraday_data, threshold_num=None):
    '''
    intraday_data: pandas dataframe of intraday data
    threshold_num: int number that requires the function to drop tickers with number of data less than the threshold
    return: pandas dataframe of cleaned intraday data
    '''
    data_length = len(intraday_data)
    # BUG: np.quantile only accept array-like real numbers
    """
    if threshold_num is None:
        q1 = np.quantile(data_length, 0.25)
        q3 = np.quantile(data_length, 0.75)
        iqr = q3-q1
        threshold_num = q1-(1.5*iqr)
    """
    tmp = intraday_data.groupby("Ticker").count().Open >= threshold_num
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
    # Reconstruct the tickers with columns less than 4681 back to 4681.
    ticker_count = cleaned_data.groupby('Ticker').count().Open
    
    # Get the full length of data
    full_length = max(ticker_count)
    
    # Find a ticker without missing data and use this dataframe as a sample
    for i in range(len(ticker_count)):
        if ticker_count[i]==full_length:
            sample_ticker = ticker_count.index[i]
            break        
    sample = cleaned_data.loc[cleaned_data['Ticker']==sample_ticker].copy()
    full_timestamps = sample.index.to_numpy()
    
    # Go though all tickers and use a expended version to reconstruct
    reconstructed_data = None
    for ticker in ticker_count.index:
        curr = cleaned_data.loc[cleaned_data['Ticker']==ticker].copy()
        
        # If this ticker has missing data, reconstruct it
        if len(curr)!=full_length:
            missing_data = curr
            curr = sample.copy()
            for timestamp in full_timestamps:
                # check every timestamp, if not missing, replace the sample row with it
                if timestamp in missing_data.index:
                    curr.loc[timestamp] = missing_data.loc[timestamp]
                # If it is missing, set all value of sample to None, set current tickername
                else:
                    curr.loc[timestamp,curr.columns[1:]]=None
                    curr.loc[timestamp,curr.columns[0]] = ticker
        # add reconstructed data frame to the result
        if reconstructed_data is not None:
            reconstructed_data = pd.concat([reconstructed_data,curr])
        else:
            reconstructed_data = curr         
    return reconstructed_data

def impute_missing_data (reconstructed_data):
    '''
    reconstructed_data: pandas dataframe of cleaned intraday data after reconstructed with missing None value
    return: pandas dataframe of imputed intraday data, missing value is linearly interpolated
    '''
    # Impute missing data for the reconstructed tickers.
    tickernames = reconstructed_data['Ticker'].unique()
    for ticker in tickernames:
        curr = reconstructed_data.loc[reconstructed_data['Ticker']==ticker]
        
        # If the ticker contains missing data, linear interpolate it
        if curr.iloc[:,1].isnull().any():
            curr.iloc[:,1:]  = curr.iloc[:,1:].interpolate(method='linear')
            reconstructed_data.loc[reconstructed_data['Ticker']==ticker] = curr
            
    # If there are some data can't be interpolated, e.g. missing data at the start or the end of the period
    # Remove the ticker since interpolate won't give good guess
    tickers_to_remove = reconstructed_data[reconstructed_data.isnull().any(axis=1)].Ticker.unique()
    imputed_data = reconstructed_data[~reconstructed_data['Ticker'].isin(tickers_to_remove)]
    return imputed_data

"""
# Clean the data and save in local
intraday_data = pd.read_pickle('russell_1000_intraday_data.pkl')
threshold_num = 4600
cleaned_intraday_data = drop_ticker (intraday_data, threshold_num)
reconstructed_intraday_data = reconstruct_dataframe(cleaned_intraday_data)
reconstructed_intraday_data.to_pickle("reconstructed_intraday_data.pkl")
imputed_intraday_data = impute_missing_data(reconstructed_intraday_data)
imputed_intraday_data.to_pickle("imputed_intraday_data.pkl")

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