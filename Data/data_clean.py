import pandas as pd

#################################### Data Adjustment ########################################################
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

def reconstruct_dataframe(cleaned_intraday_data):
    # Reconstruct the tickers with columns less than 4680 back to 4680.
    return
def impute_missing_data (cleaned_intraday_data):
    # Impute missing data for the reconstructed tickers.
    return

intraday_data = pd.read_pickle('russell_1000_intraday_data.pkl')
threshold_num = 4600
cleaned_intraday_data = drop_ticker (intraday_data, threshold_num)
print()

#################################### Volatility ########################################################
def realized_vol (cleaned_intraday_data):
    # calculate daily realized volatility
    return


