import pandas as pd
import numpy as np

#################################### Data Adjustment ########################################################
def drop_ticker (intraday_data, threshold_num=None):
    '''
    intraday_data: pandas dataframe of intraday data
    threshold_num: int number that requires the function to drop tickers with number of data less than the threshold
    return: pandas dataframe of cleaned intraday data
    '''
    data_length = intraday_data.groupby('Ticker').count().Open
    if threshold_num is None:
        q1 = np.quantile(data_length, 0.25)
        q3 = np.quantile(data_length, 0.75)
        iqr = q3-q1
        threshold_num = q1-(1.5*iqr)
    tmp = data_length >= threshold_num
    ticker_name = tmp.loc[tmp==True].index
    cleaned_intraday_data = intraday_data.reset_index().set_index('Ticker').loc[ticker_name]
    cleaned_intraday_data.rename(columns={'index':'timestamp'}, inplace=True)

    return cleaned_intraday_data.reset_index().set_index('timestamp')

def reconstruct_dataframe(cleaned_data):
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

intraday_data = pd.read_pickle('russell_1000_intraday_data.pkl')
cleaned_intraday_data = drop_ticker (intraday_data)
print()

#################################### Volatility ########################################################
def realized_vol (cleaned_intraday_data):
    # calculate daily realized volatility
    return


