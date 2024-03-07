
import pandas as pd
from fbprophet import Prophet


sales_data = pd.read_csv('sales_data.csv')


sales_data.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)


model = Prophet()
model.fit(sales_data)


future_dates = model.make_future_dataframe(periods=365)  


forecast = model.predict(future_dates)

fig = model.plot(forecast)
