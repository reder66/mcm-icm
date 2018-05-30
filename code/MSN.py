import os
from datetime import date
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation


for root, dirs, files in os.walk('MSN_data'):
    msns = files

for msn in msns:
    df = pd.read_csv('MSN_data/' + msn)
    ds = df['ds']
    df['ds'] = [str(y)+'-1-1' for y in df['ds']]

    m = Prophet()
    m.add_seasonality(
        name='financial_crisis',
        period=25*365.25,
        fourier_order=3)
    m.fit(df)
    future = m.make_future_dataframe(periods=42, freq='Y')
    forecast = m.predict(future)

    fig = m.plot(forecast)
    fig.savefig('MSN_out/fig/prediction_{0}.png'.format(msn.split('.')[0]))
    forecast['ds'] = [str(y)[0:4] for y in forecast['ds']]
    forecast.to_csv('MSN_out/csv/prediction_{0}'.format(msn))

