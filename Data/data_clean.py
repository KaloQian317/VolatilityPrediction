import pandas as pd

def drop_ticker (intraday_data, threshold_num):
    '''
    intraday_data: pandas dataframe of intraday data
    threshold_num: int number that requires the function to drop tickers with number of data less than the threshold
    return: pandas dataframe of cleaned intraday data
    '''
    tmp = intraday_data.groupby("Ticker").count().Open >= threshold_num
    ticker_name = tmp.loc[tmp==True].index
    cleaned_intraday_data = intraday_data.reset_index().set_index('Ticker').loc[ticker_name]
    cleaned_intraday_data.rename(columns={'index':'timestamp'}, inplace=True)

    return cleaned_intraday_data.reset_index().set_index('timestamp')

intraday_data = pd.read_pickle('russell_1000_intraday_data.pkl')
threshold_num = 4600
drop_ticker (intraday_data, threshold_num)
