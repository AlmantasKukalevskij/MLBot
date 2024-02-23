import yfinance as yf

# Download historical data for TTWO
ttwo_data = yf.download("GOOG", start="2020-01-01", end="2021-12-31")

# Save the data to a CSV file
ttwo_data.to_csv("GOOG_2020.csv")