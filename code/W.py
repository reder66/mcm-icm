from datetime import date
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation

results = {}
results_rate = {}

for state in ['az', 'ca', 'nm', 'tx']:
    df = pd.read_csv('W_data/w_{0}.csv'.format(state))
    df['ds'] = [str(y)+'-1-1' for y in df['ds']]

    m = Prophet()
    m.add_seasonality(
        name='financial_crisis',
        period=25*365.25,
        fourier_order=2)
    m.fit(df)
    future = m.make_future_dataframe(periods=42, freq='Y')
    forecast = m.predict(future)

    df_cv = cross_validation(m,
            horizon='365.25 days',
            initial='10958 days')
    print('#'*79)
    result = [row['yhat_lower'] <= row['y'] <= row['yhat_upper'] \
            for index, row in df_cv.iterrows()]
    results[state] = result
    results_rate[state] = result.count(True) / len(result)

    fig = m.plot(forecast)
    fig.savefig('W_out/fig/prediction_{0}.png'.format(state))
    forecast['ds'] = [str(y)[0:4] for y in forecast['ds']]
    forecast.to_csv('W_out/csv/prediction_{0}.csv'.format(state))


for state in ['az', 'ca', 'nm', 'tx']:
    print('{0}: {1}'.format(state, results_rate[state]))

