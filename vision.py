import datetime
import math
import numpy as np
import datetime as dt
import uuid
import json
import time

import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import rmse

import prophet

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

# Import the necessary SQLAlchemy modules

from sqlalchemy.orm import Session
from sqlalchemy import text, desc, asc, func


# DEPRECATED
import sqlite3 # to be evicted to visiondata
import pandas as pd # do we need this here??
import visiondata as vd


class Predictor:

    @staticmethod
    def db_load(db_id):
        # engine = create_engine('sqlite:///instance/default.db')

        engine = vd.DataStorage.connect()

        sess = Session(engine)

        p = sess.query(vd.DataStorage.PredictorModel).filter_by(id=db_id).first()

        # datasource
        datasource = json.loads(p.datasource)
        ds = vd.DataSource.from_json(datasource)
        series = vd.Series(datasource=ds)

        # model
        model = json.loads(p.model)
        type = model['type']
        params = model['params']

        pred = None
        if type == 'sarimax':
            order = [int(x) for x in params['order']]
            sorder = [int(x) for x in params['sorder']]
            pred = SARIMAXPredictor(series, p.ts_now, order=order, sorder=sorder)
        elif type == 'prophet':
            pred = ProphetPredictor(series, p.ts_now)
            pred.n_history = 30
        elif type == 'lstm':
            pred = LSTMPredictor(series, p.ts_now)
        pred.name = p.title
        pred.set_hparams(params)

        pass

        pred.guid = p.guid
        vd.DataStorage.close(engine)
        return pred

    @staticmethod
    def from_db_model(model):
        series = vd.Series.from_json("model.datasource")
        pred = Predictor(series, datetime.datetime(year=2023, month=10, day=15) )
        pred.guid = model.guid
        return pred

    # def to_db_model(self, model):
    #     model.datasource = str(self.series)
    #     model.guid = self.guid
    #     model.ts_now = self.ts_now
    #     return model

    def save(self):
        engine = vd.DataStorage.connect()
        sess = Session(engine)
        p = sess.query(vd.DataStorage.PredictorModel).filter_by(guid=self.guid).first()
        p.ts_now = self.ts_now

        sess.commit()
        sess.close()

        self.storage_close(engine)

    def __init__(self, series, ts_now):
        self.id_item = 0
        self.guid = str(uuid.uuid4())
        self.model = None
        self.fit_model = None
        self.series = series

        self.n_forecast = 1
        self.n_period = 24
        self.n_history = 30

        self.n_X = self.n_history * self.n_period
        self.n_y = self.n_forecast * self.n_period

        self.ts_history = None
        self.ts_now = None
        self.ts_future = ts_now
        self.timestep()

    def __hash__(self):
        return self.guid

    def hash(self):
        return str(self.guid)

    def __str__(self):
        return f"Predictor {self.guid}"

    def storage_open(self):
        return vd.DataStorage.connect(self.guid)

    def storage_close(self, engine):
        vd.DataStorage.close(engine)

    def generate_future(self, df):
        future = df.index.shift(self.n_y)[-self.n_y:]
        return future

    def set_hparams(self, hparams):
        if 'n_history' in hparams:
            self.n_history = hparams['n_history']
        self.n_X = self.n_history * self.n_period
        print(self.n_X, self.n_history, self.n_period, self.n_history * self.n_period)
        print('Predictor', 'n_history', self.n_history, 'n_X', self.n_X)
        ...

    def get_autotune_search_space(self):
        ...

    def get_autotune_results(self, num_rows=10):
        engine = self.storage_open()
        sess = Session(engine)

        results = sess.query(vd.DataStorage.AutotuneVisionModel).where(vd.DataStorage.AutotuneVisionModel.datetime >= '2024-03-03 00:00:00').order_by(
            asc(vd.DataStorage.AutotuneVisionModel.mean_error))\


        if num_rows < 1:
            results = results.all()
        else:
            results = results.limit(num_rows)

        return results

    def autotune(self, hparams=False):
        print('Predictor', 'Autotune')
        if not hparams:
            hparams = self.get_autotune_search_space()
            print('Predictor', 'Autotune on default search space')

        print('Predictor', 'Hyperparams', hparams)
        print()
        results = [[] for i in range(len(hparams))]
        timesteps = 0
        while True:

            if timesteps > 3:
                break

            for i, hparam in enumerate(hparams):
                print('- hyperparams', hparam)
                self.set_hparams(hparam)

                self.ts_history = self.ts_now + dt.timedelta(hours=-self.n_X)
                print('Load data', (self.ts_history, self.ts_now))
                train = self.series.load_data(self.ts_history, self.ts_now)
                test = self.series.load_data(self.ts_now, self.ts_now + dt.timedelta(days=1))
                if len(test) < self.n_y:
                    print('Not enough test data!!')
                    print('Train/test sets', len(train), len(test))
                    break

                # train, test = df.iloc[:-24], df.iloc[-24:]

                # generate future (not used here)
                future = self.generate_future(train)

                start_time = time.time()

                # fit model
                self.fit(train)

                training_time = time.time() - start_time

                # predict values
                forecast = self.predict(steps=self.n_y)

                forecast = forecast.rename('yhat')
                forecast.index.name = 'x'
                test['yhat'] = forecast

                result_df = test.dropna()
                error = self._eval_predictions(result_df['y'], result_df['yhat'])
                print('Predictor autotune error', error.mean())
                results[i].append(error.mean())
                # print(error)

                engine = self.storage_open()
                sess = Session(engine)

                db = vd.DataStorage.AutotuneVisionModel()
                db.datetime = dt.datetime.now()
                db.hparams = json.dumps(hparam)
                db.training_time = training_time
                db.mean_error = error.mean()
                db.results = error.mean()
                sess.add(db)

                sess.commit()
                sess.close()

                self.storage_close(engine)





            self.timestep()
            timesteps += 1

        print('Autotune finished')
        print('Hyperparams')
        print(hparams)
        print('Mean error')
        print(results)

        results = [np.mean(errs) for errs in results]
        print(results)

        pass

    def fit(self, df):
        print('Predictor', 'fit')
        print('n_history', self.n_history)
        pass

    def predict(self, steps):
        print('Predictor', 'predict')
        pass

    def store_prediction(self, series):
        # Connect to the SQLite database (creates a new database if it doesn't exist)
        # connection = sqlite3.connect("db/mydatabase.db")
        df = pd.DataFrame(series)
        df.insert(loc=0, column='id_item', value=self.id_item)
        df = df.set_index([df.index, 'id_item'])
        # Save the DataFrame to the database
        # engine = vd.DataStorage.connect()
        engine = self.storage_open()
        df.to_sql('tmp_vision', engine, if_exists='append', index=True)

        # cur = connection.cursor()
        sess = Session(engine)

        sess.execute(text("""
            UPDATE data_vision 
            SET yhat = tmp.yhat 
                FROM tmp_vision AS tmp 
                WHERE data_vision.id_item = tmp.id_item 
                AND data_vision.x = tmp.x
        """))

        sess.execute(text("""
            UPDATE tmp_vision 
            SET yhat = NULL 
                FROM data_vision 
                WHERE data_vision.id_item = tmp_vision.id_item 
                AND data_vision.x = tmp_vision.x
        """))

        sess.execute(text("""
            DELETE FROM tmp_vision WHERE yhat IS NULL
        """))

        sess.execute(text("""
            INSERT INTO data_vision 
            (id_item, x, yhat) 
            SELECT 
                id_item, 
                x, 
                yhat 
            FROM tmp_vision
        """))

        sess.execute(text("""
            DELETE FROM tmp_vision
        """))

        sess.commit()

        sess.close()
        # Close the connection
        # vd.DataStorage.close(engine)
        self.storage_close(engine)

    def load_predictions(self, ts_from, ts_to):
        # Connect to the SQLite database (creates a new database if it doesn't exist)
        # conn = sqlite3.connect("db/mydatabase.db")
        # engine = vd.DataStorage.connect()
        engine = self.storage_open()
        # Define your SQL query
        sql_query = "SELECT x, yhat, error FROM data_vision " \
                    "WHERE id_item = " + str(self.id_item) + " AND " \
                    "      x >= '" + vd.get_datetime_str(ts_from) + "' AND " \
                    "      x < '" + vd.get_datetime_str(ts_to) + "' " \
                    "ORDER BY x ASC "

        # print("DB SELECT")
        # print(sql_query)
        # Use pandas to read the query results into a DataFrame
        df = pd.read_sql_query(sql_query, engine)
        # print("DB DONE")
        df = df.set_index('x')
        # Close the connection
        # conn.close()
        # vd.DataStorage.close(engine)
        self.storage_close(engine)

        return df

    def _eval_predictions(self, y, y_hat):
        error = np.sqrt((y - y_hat) ** 2)
        return error
        pass

    def eval_predictions(self, period=-1):
        ts_start = self.ts_now + dt.timedelta(hours=self.n_y*period)
        ts_end = self.ts_now

        df = self.series.load_data(ts_start, ts_end)
        # print('eval_predictions data load', ts_start, ts_end)
        # print(df)
        forecast = self.load_predictions(ts_start, ts_end)
        # print("Loaded forecast", forecast)
        if len(forecast) > 0:
            df['yhat'] = forecast['yhat'].values

            # print(df['y'], df['yhat'])

            # df['error'] = np.sqrt((df['y'].values - df['yhat'].values) ** 2)
            df['error'] = self._eval_predictions(df['y'].values, df['yhat'].values)
            # print('eval_predictions')
            # print(df)


            # engine = vd.DataStorage.connect()
            engine = self.storage_open()
            sess = Session(engine)

            #connection = sqlite3.connect("db/mydatabase.db")

            for index, row in df.iterrows():
                # Replace 'your_table' with the name of your table, 'db_id' with your unique key column, and 'value' with the column you want to update
                update_query = f"UPDATE data_vision SET error = {row['error']} WHERE id_item= {self.id_item} AND strftime('%s', x) = strftime('%s', '{index}')"
                # print(update_query)
                sess.execute(text(update_query))
                #connection.execute(update_query)

            sess.commit()
            sess.close()
            #vd.DataStorage.close(engine)
            self.storage_close(engine)
            # connection.commit()

            # cur.close()
            # Close the connection
            # connection.close()
            try:
                # # Step 8: Visualize the results
                # plt.figure(figsize=(12, 6))
                # # plt.plot(df.index, df, label='History')
                # plt.plot(df.index, df['y'], label='Actual')
                # plt.plot(df.index, df['yhat'], label='Forecast')
                # plt.plot(df.index, df['error'], label='Error')
                # plt.legend()
                # plt.xlabel('Date')
                # plt.ylabel('Value')
                # plt.title('Forecast '+str(self.__class__.__name__))
                # plt.show()
                #
                # rmse_value = rmse(df['yhat'], df['y'])
                # print(f"Root Mean Squared Error (RMSE): {rmse_value}")
                pass
            except Exception as e:
                print('Eval predict', e)
                pass
        else:
            print("No forecast found?!", ts_start, ts_end)

    def timestep(self):
        self.ts_now = self.ts_future

        self.ts_history = self.ts_now + dt.timedelta(hours=-self.n_X)
        self.ts_future = self.ts_now + dt.timedelta(hours=self.n_y)

    def step(self):
        print("Predictor", "step()", "")
        if self.ts_now >= dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
            print('We\'re done for now ...')
            return False
        try:
            # load data
            print("Predictor", "step()", "load data")
            df = self.series.load_data(self.ts_history, self.ts_now)

            # generate future (not used here)
            print("Predictor", "step()", "generate future")
            future = self.generate_future(df)

            # fit model
            print("Predictor", "step()", "fit model")
            self.fit(df)

            # predict values
            print("Predictor", "step()", "make forecast")
            forecast = self.predict(steps=self.n_y)
            # print('Predictor forecast', forecast)
            forecast = forecast.rename('yhat')
            forecast.index.name = 'x'

            print("Predictor", "step()", "store predictions")
            self.store_prediction(forecast)

            print("Predictor", "step()", "eval predictions")
            self.eval_predictions()

            # # evaluate and visualise
            # # @TODO Not really supposed to be here, right...
            # future_vals = self.series.load_data(self.ts_now, self.ts_future)
            # actual_values = future_vals['y'][:self.n_y]
            # rmse_value = rmse(forecast, actual_values)
            # print(f"Root Mean Squared Error (RMSE): {rmse_value}")

            return True
        except Exception as e:
            print('Step', e)
            pass
        finally:
            # do a time step
            print("Predictor", "step()", "advance time")
            self.timestep()

            print("Predictor", "step()", "save state")
            # update predictor db record
            self.save()
            pass

    def run(self):
        print('Predictor - run()')
        # while self.step():
        #     pass
        print('DEBUG - Predictors work one step at a time')
        self.step()

class LSTMPredictor(Predictor):

    def __init__(self, series, ts_now, n_timesteps=24, n_samples=24, n_layers=1, n_neurons=32, n_epochs=5):
        super().__init__(series, ts_now)

        self.scaler = None
        self.model = None

        self.n_forecast = n_samples
        self.n_period = n_timesteps
        self.n_history = 30

        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_epochs = n_epochs
        self.batch_size = 1

        self.verbose = 1
        self._init_model()

    def __str__(self):
        return f"Predictor LSTM {self.n_layers}x{self.n_neurons} {self.n_epochs}E / {self.n_history}"

    def set_hparams(self, hparams):
        super().set_hparams(hparams)
        if 'n_layers' in hparams:
            self.n_layers = hparams['n_layers']
        if 'n_neurons' in hparams:
            self.n_neurons = hparams['n_neurons']
        if 'n_epochs' in hparams:
            self.n_epochs = hparams['n_epochs']
        if 'batch_size' in hparams:
            self.batch_size = hparams['batch_size']

        print('n_layers', self.n_layers)
        print('n_neurons', self.n_neurons)
        print('n_epochs', self.n_epochs)
        print('batch_size', self.batch_size)

    def _init_model(self, **kwargs):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(1, self.n_period)))
        for _ in range(self.n_layers):
            self.model.add(LSTM(self.n_neurons, input_shape=(self.n_period,), return_sequences=True))
            self.model.add(Dense(self.n_period))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def _scale_data(self, data):
        # fit scaler
        if not self.scaler:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler = self.scaler.fit(data.values)

        data_scaled = self.scaler.transform(data.values)
        return data_scaled

    def _inv_scale_data(self, data):
        if not self.scaler:
            return False
        return self.scaler.inverse_transform(data)

    # frame a sequence as a supervised learning problem
    def _to_supervised(self, data, lags=1):
        df = pd.DataFrame(
            [data[i:i + (self.n_period + 1)].flatten() for i in range(len(data) - (self.n_period + 1))])
        return np.asarray(df)
        pass

    def fit(self, train):
        super().fit(train)
        self._init_model()
        self.train = train
        train = self._scale_data(train)
        sup = self._to_supervised(train, self.n_period)
        X, y = sup[:, 0:-1], sup[:, -1:]
        X = X.reshape(X.shape[0], 1, X.shape[1])

        callbacks = [
            # EarlyStopping(monitor='loss', patience=10)
        ]

        history = self.model.fit(X, y, epochs=self.n_epochs, verbose=self.verbose, callbacks=callbacks)
        print('LSTM Training history')
        print(history.history)
        pass

    def predict(self, steps):
        train = self.train
        scaled = self._scale_data(train)
        X = scaled[-self.n_period:]
        future = pd.date_range(start=train.index[-1], periods=self.n_forecast + 1, freq='H')[1:]
        df = pd.DataFrame(future, columns=["ds"])
        df = df.set_index('ds')
        X = np.asarray(X).reshape(1, 1, self.n_period)
        predictions = []
        for t in range(self.n_forecast):
            forecast = self.model.predict(X, verbose=self.verbose)
            predictions.append(forecast)
            X = np.append(X, forecast, axis=2)
            X = np.delete(X, 0, axis=2)
        predictions = np.asarray(predictions).reshape(-1, 1)
        predictions = self._inv_scale_data(predictions)
        df['yhat'] = predictions
        return df['yhat']
        pass

    def get_autotune_search_space(self):

        a_ly = [1, 2, 3, 4]
        a_ne = [16, 32, 64, 128]
        a_ep = [50, 100, 300]

        a_ba = [1]
        a_n = [30, 60, 90]


        # a_ly = [1]
        # a_ne = [64]
        # a_ep = [500]
        # a_n = [30, 60, 90, 365]
        hparams = []

        for n in a_n:
            for ly in a_ly:
                for ne in a_ne:
                    for ep in a_ep:
                        for ba in a_ba:
                            hparams.append({
                                'n_history': n,
                                'n_layers': ly,
                                'n_neurons': ne,
                                'n_epochs': ep,
                                'batch_size': ba
                            })

        return hparams

        return []


class SARIMAXPredictor(Predictor):

    def __init__(self, series, ts_now, order=(1, 1, 1), sorder=(1, 1, 1, 24)):
        # order = (p, d, q)
        # p = 1  # Autoregressive order
        # d = 1  # Differencing order
        # q = 1  # Moving average order
        # sorder = (P, D, Q, s)
        # P = 1  # Seasonal autoregressive order
        # D = 1  # Seasonal differencing order
        # Q = 1  # Seasonal moving average order
        # s = 24  # Seasonal period (e.g., 12 for monthly data)
        super().__init__(series, ts_now)
        self.id_item = 1
        self.order = order
        self.sorder = sorder

    def __str__(self):
        return f"Predictor SARIMA {self.order} {self.sorder} / {self.n_history}"

    def set_hparams(self, hparams):
        super().set_hparams(hparams)
        self.order = hparams['order']
        self.sorder = hparams['sorder']

        print('order', self.order)
        print('sorder', self.sorder)

    def fit(self, df):
        super().fit(df)
        print('SARIMAX FIT', 'len(df)', len(df))
        print(df.index, df['y'])
        try:
            self.model = SARIMAX(df['y'], order=self.order, seasonal_order=self.sorder,
                                 initialization='approximate_diffuse')
            self.fit_model = self.model.fit()
        except Exception as e:
            print('Fit', e)
            return False
        return self.fit_model

    def predict(self, steps):
        try:
            forecast = self.fit_model.get_forecast(steps=steps)
        except Exception as e:
            print('Forecast', e)
            return False
        return forecast.predicted_mean

    @staticmethod
    def get_autotune_search_space():
        # hparams = [
        #     {
        #         'n_history': 30,
        #         'order': [1, 0, 1],
        #         'sorder': [2, 0, 1, 24]
        #     },
        #     {
        #         'n_history': 60,
        #         'order': [1, 0, 1],
        #         'sorder': [2, 0, 1, 24]
        #     },
        #     {
        #         'n_history': 168,
        #         'order': [1, 0, 1],
        #         'sorder': [2, 0, 1, 24]
        #     },
        # ]
        #
        # a_p = [1, 2, 3]
        # a_d = [0]
        # a_q = [0, 1, 2, 3]
        #
        # a_sp = [0, 1, 2]
        # a_sd = [0]
        # a_sq = [0, 1, 2]
        #
        # a_sm = [24, 48]
        #
        # a_n = [30, 60, 90]

        # a_p = [1, 2, 3]
        # a_d = [0]
        # a_q = [0, 1, 2, 3]
        #
        # a_sp = [1, 2]
        # a_sd = [0]
        # a_sq = [0, 1, 2]
        #
        # a_sm = [24, 48]
        #
        # a_n = [30, 60, 90]


        a_p = [2, 3]
        a_d = [0]
        a_q = [0, 1, 2, 3]

        a_sp = [2]
        a_sd = [0]
        a_sq = [0, 1, 2]

        a_sm = [24, 48]

        a_n = [90]


        orders = [(p, d, q) for p in a_p for d in a_d for q in a_q]
        sorders = [(sp, sd, sq, sm) for sp in a_sp for sd in a_sd for sq in a_sq for sm in a_sm]

        orders = [(2,0,1), (2,0,2), (3,0,0), (2,0,0)]

        hparams = []

        for n in a_n:
            for o in orders:
                for so in sorders:
                   hparams.append({
                       'n_history': n,
                       'order': o,
                       'sorder': so
                   })

        print(len(hparams))

        return hparams


class ProphetPredictor(Predictor):

    def __init__(self, series, ts_now, n_timesteps=1, n_samples=1):
        super().__init__(series, ts_now)
        self.id_item = 2
        self.model_params = dict({"daily_seasonality": True,
                                  "weekly_seasonality": True,
                                  "yearly_seasonality": False,
                                  "seasonality_mode": "additive",
                                  "growth": "linear"})

        self._init_model()

    def __str__(self):
        return f"Predictor Prophet {self.model_params['daily_seasonality'] and 'D ' or ''}" \
               f"{self.model_params['weekly_seasonality'] and 'W ' or ''}" \
               f"{self.model_params['yearly_seasonality'] and 'Y ' or ''}" \
               f"{self.model_params['seasonality_mode']} " \
               f"{self.model_params['growth']}"

    def set_hparams(self, hparams):
        super().set_hparams(hparams)
        if 'model_params' in hparams:
            self.model_params = hparams['model_params']
        print('model_params', self.model_params)
    def _init_model(self, **kwargs):
        self.model = prophet.Prophet(**self.model_params)

    def fit(self, train):
        super().fit(train)
        self._init_model()
        df = train.reset_index()

        df.columns = ['ds', 'y']

        self.model.fit(df)
        pass

    def predict(self, steps):
        # vytvoříme časovou řadu (index) pro predikci
        future = self.model.make_future_dataframe(periods=steps, freq='H')

        # (pro logistický trend je potřeba definovat kapacitu)
        # future['cap'] = data['cap'].max()

        # napredikujeme poždaované hodnoty
        forecast = self.model.predict(future)
        forecast = forecast.set_index('ds')
        return forecast['yhat'][-steps:]

    def get_autotune_search_space(self):

        a_n = [30, 60, 90]


        hparams = []
        for d in [True, False]:
            for w in [True, False]:
                for y in [True, False]:
                    for g in ['linear']:
                        for m in ['additive', 'multiplicative']:
                            for n in a_n:

                                hparams.append({
                                    'n_history': n,
                                    'model_params': {
                                        "daily_seasonality": d,
                                        "weekly_seasonality": w,
                                        "yearly_seasonality": y,
                                        "seasonality_mode": m,
                                        "growth": g
                                    }
                                })

        print(len(hparams))

        return hparams


class Detector:

    def __init__(self, predictor, ts_from=False, ts_to=False):
        self.predictor = predictor

        # setup time frame
        if not ts_to or ts_to == 'False':
            ts_to = dt.datetime(year=2021, month=4, day=1)
            ts_to = self.predictor.ts_now
        if not ts_from or ts_from == 'False':
            ts_from = dt.datetime(year=2021, month=1, day=1)
            ts_from = self.predictor.ts_history

        # ts_from = ts_to - dt.timedelta(days=30)

        # load actual data from datasource
        y = self.predictor.series.load_data(ts_from, ts_to)
        y.set_index(pd.to_datetime(y.index), inplace=True)

        # load forecast
        forecast = self.predictor.load_predictions(ts_from, ts_to)
        forecast.set_index(pd.to_datetime(forecast.index), inplace=True)

        # assemble dataframe
        df = pd.DataFrame(y)
        df['yhat'] = forecast.yhat
        df['error'] = forecast.error

        self.df = df
        pass

    def analyze(self):

        # define rolling windows for stat calculations
        roll_8 = self.df.rolling(window=8)
        roll_24 = self.df.rolling(window=24)

        #
        self.df['diff'] = self.df['y'] - self.df['yhat']
        mean = self.df['y'].mean()
        std = self.df['y'].std()
        # self.df['analyze_zscore'] = (self.df['y'] - mean) / std
        # self.df['analyze_zscore_24'] = self.df['analyze_zscore'].rolling(window=6).mean()

        self.df['y_mean_8'] = roll_8['y'].mean()
        self.df['y_mean_24'] = roll_24['y'].mean()
        self.df['y_std_8'] = roll_8['y'].std()
        self.df['y_std_24'] = roll_24['y'].std()
        self.df['y_sum_8'] = roll_8['y'].sum()
        self.df['y_sum_24'] = roll_24['y'].sum()

        y_mean = self.df['y'].mean()
        y_std = self.df['y'].std()
        self.df['y_std'] = y_std
        self.df['y_2std'] = y_std * 2
        self.df['y_3std'] = y_std * 3
        self.df['y_mean'] = y_mean
        self.df['y_q1'] = y_mean + y_std

        self.df['y_q2'] = y_mean + y_std * 2
        self.df['y_q3'] = y_mean + y_std * 3
        self.df['diff_mean'] = self.df['diff'].mean()
        self.df['diff_std'] = self.df['diff'].std()
        self.df['diff_mean_8'] = roll_8['diff'].mean()
        self.df['diff_mean_24'] = roll_24['diff'].mean()

        self.df['diff_cumm_8'] = roll_8['diff'].sum()
        self.df['diff_cumm_24'] = roll_24['diff'].sum()

        self.df['error_mean'] = self.df['error'].mean()
        self.df['error_std'] = self.df['error'].std()

        self.df['error_mean_8'] = roll_8['error'].mean()
        self.df['error_mean_24'] = roll_24['error'].mean()

        self.df['error_cumm_8'] = roll_8['error'].sum()
        self.df['error_cumm_24'] = roll_24['error'].sum()

        self.df['alarm_level'] = self.df['y_mean_24'] + 1.0 * self.df['y_std_24']
        self.df['alarm'] = self.df['diff_cumm_24'] > (self.df['alarm_level'])
        self.df['diff_cumm_24_rel'] = self.df['diff_cumm_24'] / self.df['error_mean']

        # compare error

        #do some rolling averages

        # compare those

        # do some magic
        # error_mean_8 > y_std_24??

        # raturn incidents
        # print(self.df[self.df['error'] > self.df['y_std_24']])

        # fig = self.df.plot()
        # fig.show()
        return self.df.copy()
        pass