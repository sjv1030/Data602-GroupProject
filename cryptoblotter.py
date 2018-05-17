import datetime
import time
import pandas as pd
import requests
import io
from io import BytesIO
from flask import Flask, session, render_template, request, send_file, make_response, url_for
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly.graph_objs import *
import json
from py_scripts import get_data, py_mongo, LSTM_results


app = Flask(__name__)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

@app.route("/")
def template():
    list_of_futures = ["Oil", "Natural Gas"]

    blotter = get_data.generate_dataframe(list_of_futures)
    session['blotter'] = blotter.to_json()
    return render_template('index.html', vars = [blotter, list_of_futures], tables = [blotter.to_html()])

@app.route("/numshares", methods = ['POST'])
def get_future_name():
    future = '{}'.format(request.form['future'])
    blotter_json = pd.read_json(session['blotter'])
    blotter = pd.DataFrame(blotter_json)
    future_chosen = str(future)
    session['future_value'] = future_chosen

    graphs = [
        dict(
            data=[{
                    'x': df.index,
                    'y': df[col],
                    'name': col
                }  for col in df.columns],
            layout=dict(title='Total Usage Graph')
        ),
        dict(
            data=[{
                    'x': df.index,
                    'y': df['close']
                }],
            layout=dict(title='Average Hourly Usage')
        )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('stats.html', ids=ids, graphJSON=graphJSON, var = [crypto, blotter], tables = [stats.to_html(), blotter.to_html()])

@app.route("/startup", methods = ['POST'])
def get_shares_sell():
    crypto_chosen = session['crypto_value']
    blotter_json = pd.read_json(session['blotter'])
    blotter = pd.DataFrame(blotter_json)
    data_24_hours = get_data.price_24_hours(crypto_chosen, "USD")
    stats = get_data.crypto_analytics(data_24_hours, crypto_chosen)
    # get value
    model_type_val = request.form['model_type']

    if model_type_val == 'LSTM':
        basic_df, yhat_inverse = LSTM_results.run_LSTM(crypto_chosen)
    else:
        basic_df, yhat_inverse = LSTM_results.run_LSTM(crypto_chosen)

    graphs = [
        dict(
            data=[{
                    'x': basic_df.index,
                    'y': basic_df['high']
                }],
            layout=dict(title='High Values')
        ),
        dict(
            data=[{
                    'x': basic_df.index,
                    'y': basic_df['low']
                }],
            layout=dict(title='Low Values')
        )
    ]

    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('numshares.html', ids=ids, graphJSON=graphJSON, var = [crypto_chosen, blotter], tables = [stats.to_html(), blotter.to_html()])

if __name__ == '__main__':
    app.run(host = '0.0.0.0')
