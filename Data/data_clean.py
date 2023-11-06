import pandas as pd

intraday_data = pd.read_pickle('russell_1000_intraday_data.pkl')
sum(intraday_data.groupby("Ticker").count().Open < 4680)
intraday_data