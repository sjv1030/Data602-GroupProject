import pandas as pd
import quandl
import json
from urllib.request import urlopen
from py_scripts import RNN_results, LSTM_results


"""
Created on Tue May 15 22:42:49 2018
@author: sjv1030_hp
"""

## This script pulls data from EIA.gov API and from Quandl.com
## Spot nat gas and wti (oil) prices are available on a daily, weekly, and monthly basis
## Two dataframes ('oil_data' & 'ng_data') are curated with monthly fundamental and economic data
## Quarterly real GDP is also available in 'gdpr'

def wti_spot_prices():
    ########### WTI Spot Prices ###############
    wti_d = quandl.get('EIA/PET_RWTC_D') # daily
    wti_w = quandl.get('EIA/PET_RWTC_W') # weekly
    wti_m = quandl.get('EIA/PET_RWTC_M') # monthly
    return (wti_d, wti_w, wti_m)

def nat_gas_spot_prices():
    ########## Nat Gas Spot Prices ############
    ng_d = quandl.get('EIA/NG_RNGWHHD_D') # daily
    ng_w = quandl.get('EIA/NG_RNGWHHD_W') # weekly
    ng_m = quandl.get('EIA/NG_RNGWHHD_M') # monthly
    return (ng_d, ng_w, ng_m)

def oil_info():
    ########### Monthly Rigs  #################
    oil_rig_id = 'PET.E_ERTRRO_XR0_NUS_C.M'
    ng_rig_id = 'PET.E_ERTRRG_XR0_NUS_C.M'
    ####### Monthly Oil Production ############
    oil_prod_id = 'PET.MCRFPUS1.M' # Field production
    oil_import_id = 'PET.MCRIMUS1.M' # Imports
    ######## Monthly Oil Inventory ############
    ## Consumption (demand) is hard to get from EIA. Additionally, we care more
    ## about the supply/demand difference. This can be found in the monthly change
    ## in inventory. If supply > demand, then inventory rises. Converse is true.
    oil_inv_id = 'PET.MCESTUS1.M'

    oil_ids = {'rig_id':'PET.E_ERTRRO_XR0_NUS_C.M',
        'prod_id':'PET.MCRFPUS1.M',
        'import_id':'PET.MCRIMUS1.M',
        'inv_id':'PET.MCESTUS1.M',
        }

    ## set api and get data
    api = 'd88caf37dd5c4619bad28016ca4f0379'
    url = 'http://api.eia.gov/series/?api_key='+api+'&series_id='

    ## Dictionaries to save data
    oil_data_dict = {}
    for k, v in oil_ids.items():
        dat = urlopen(url+v).read()
        data = json.loads(dat.decode())

        df = pd.DataFrame(data['series'][0]['data'],
                           columns=['Date',k[:-3]])
        df.set_index('Date',drop=True,inplace=True)
        df.sort_index(inplace=True)
        oil_data_dict[k[:-3]] = df

        ## Create dataframe combining all monthly data series
    oil_data = pd.DataFrame()

    for v in oil_data_dict.values():
        oil_data = pd.concat([oil_data, v], axis=1)

    oil_data['tot_supply'] = oil_data['prod'] + oil_data['import']
    oil_data['netbal'] = oil_data['inv'].diff() ## change in inventory is what we're after
    oil_data.dropna(inplace=True)
    oil_data.index = pd.to_datetime(oil_data.index+'01')
    oil_data.index = oil_data.index.to_period('M').to_timestamp('M')

    oil_data = pd.concat([oil_data,wti_m], join='inner', axis=1)
    oil_data.rename(columns={'Value':'wti'},inplace=True)
    oil_data = pd.concat([oil_data, econ_m], join='inner', axis=1)

    return oil_data

def ng_info():
    ###### Monthly Nat Gas Production #########
    ng_prod_id = 'NG.N9070US2.M'
    ###########################################

    ##### Monthly Nat Gas Consumption #########
    ng_cons_id = 'NG.N9140US1.M'
    ###########################################

    ########## US Economic Data ################
    gdpr = quandl.get('FRED/GDPMC1') # quarterly US real GDP index
    twd_m = quandl.get('FRED/TWEXBMTH') # monthly trade-weighted dollar index
    twd_d = quandl.get('FRED/DTWEXB') # daily trade-weighted dollar index
    ip = quandl.get('FRED/IPB50001N') # monthly US industrial production
    econ_m = pd.concat([twd_m, ip], join='inner', axis=1)
    econ_m.columns = ['twd','ip']
    econ_m.index = econ_m.index.to_period('M').to_timestamp('M')
    ###########################################


    ## Dictionaries to loop through when pulling down data
    ng_ids = {
            'rig_id':'PET.E_ERTRRG_XR0_NUS_C.M',
            'prod_id':'NG.N9070US2.M',
            'cons_id':'NG.N9140US1.M',
            }

    ## Dictionaries to save data
    ng_data_dict = {}

    ## set api and get data
    api = 'd88caf37dd5c4619bad28016ca4f0379'
    url = 'http://api.eia.gov/series/?api_key='+api+'&series_id='

    for k, v in ng_ids.items():
        dat = urlopen(url+v).read()
        data = json.loads(dat.decode())

        df = pd.DataFrame(data['series'][0]['data'],
                           columns=['Date',k[:-3]])

        df.set_index('Date',drop=True,inplace=True)
        df.sort_index(inplace=True)

        # Make nat gas prod same units as nat gas consumption -- billion cubic feet
        if k[:-3] == 'prod':
            df = df/1000
        ng_data_dict[k[:-3]] = df

    ## Create dataframe combining all monthly data series
    ng_data = pd.DataFrame()
    for v in ng_data_dict.values():
        ng_data = pd.concat([ng_data, v], axis=1)

    ng_data.dropna(inplace=True)
    ng_data['netbal'] = ng_data['prod'] - ng_data['cons']
    ng_data.index = pd.to_datetime(ng_data.index+'01')
    ng_data.index = ng_data.index.to_period('M').to_timestamp('M')

    ng_data = pd.concat([ng_data,ng_m], join='inner', axis=1)
    ng_data.rename(columns={'Value':'nat_gas'},inplace=True)
    ng_data = pd.concat([ng_data, econ_m], join='inner', axis=1)

    return ng_data

def get_current_price(future):
    if future == "Oil":
        wti_d, wti_w, wti_m = wti_spot_prices()
        value = wti_d.reset_index()["Value"].iloc[-1]
    else:
        ng_d, ng_w, ng_m = nat_gas_spot_prices()
        value = ng_d.reset_index()["Value"].iloc[-1]

    return value

def generate_dataframe(list_of_futures):
    """
    generates initial dataframe of values for the blotter
    this is where we run our analytics for predicting value, with an end value red in
    whenever we add in a new column, have to include in the following
    - cols
    - temp_blotter
    """
    # setup dataframe
    cols = ['Future', 'Current Price', "ARIMA", "RNN"]
    blotter = pd.DataFrame(columns = cols)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # loop through function get_price
    for future in list_of_futures:
        Future = future
        Current_Price = get_current_price(future)

        # here is where we would add more values to the dataframe for our predictions
        ARIMA = 0
        RNN = 0
        temp_blotter = pd.Series([Future, Current_Price, ARIMA, RNN])
        df_blotter = pd.DataFrame([list(temp_blotter)], columns = cols)
        blotter = blotter.append(df_blotter)

    blotter = blotter.set_index('Future')
    return blotter

def analytics(df, crypto):
    """
    calculate min, max, standard deviation, mean
    """
    price = df['close'].values
    min_val = min(price)
    max_val = max(price)
    num_items = len(price)
    mean = sum(price) / num_items
    differences = [x - mean for x in price]
    sq_differences = [d ** 2 for d in differences]
    ssd = sum(sq_differences)
    variance = ssd / num_items
    sd = round(sqrt(variance), 2)
    avg_price_crypto = avg_price(crypto, "USD")
    d = {"cryptocurrency": [crypto], "min": [min_val], "max": [max_val], "sd": [sd], "avg price": [avg_price_crypto]}
    stats = pd.DataFrame(data = d).set_index('cryptocurrency')
    return stats

def as_float(number):
    """
    converts currency values to floats for doing math

    ex: $5,000.00 -> 5000.00
    """
    number = str(number)
    if number != 0 and ',' in number:
        number = number.replace(",","")
        number = float(number.replace("$",""))
    elif number != 0:
        number = float(number.replace("$",""))
    return number

def clean_data (blotter, index, column):
    """
    converts all currency values to floats for doing math in the blotter

    ex: $5,000.00 -> 5000.00
    """
    value = str(blotter.loc[index, column])
    if value != 0:
        value = value.replace(",","")
        value = float(value.replace("$",""))
    return(value)

def as_currency(amount):
    """
    converts numeric-like values to currency for display

    ex: 5000.00 -> $5,000.00
    """
    if type(amount) == str:
        return '${:,.2f}'.format(amount)
    elif amount >= -10000000:
        return '${:,.2f}'.format(amount)
