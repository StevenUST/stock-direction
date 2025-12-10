import yfinance as yf

ticker = "AAPL"
start_date = "2021-01-01"
end_date = "2024-12-31"

data = yf.download(ticker, start=start_date, end=end_date)

data.to_csv("data/stock_data.csv")