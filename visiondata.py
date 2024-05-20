import os

import pandas as pd
import psycopg2
import sqlite3
from datetime import datetime, timedelta
import datetime as dt

#@TODO Does it really have to be here?!
def get_datetime_str(dt):

    if type(dt) != str:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt

# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.pool import QueuePool

# Import the necessary SQLAlchemy modules
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base


class DataStorage:

    DefaultBase = declarative_base()
    DataBase = declarative_base()

    class PredictorModel(DefaultBase):
        __tablename__ = "predictor"
        id = Column(Integer, primary_key=True)
        guid = Column(String(36), nullable=False)
        title = Column(String(200), nullable=False)
        datasource = Column(Text, nullable=False)
        model = Column(Text, nullable=False)
        ts_now = Column(DateTime, nullable=False)
        enabled = Column(Boolean, nullable=False, default=True)

    class JobModel(DefaultBase):
        __tablename__ = "job"
        guid = Column(String(36), primary_key=True)
        desc = Column(Text, nullable=False)
        pid = Column(Integer, nullable=True)
        started = Column(DateTime, nullable=False, default=dt.datetime.utcnow)
        updated = Column(DateTime, nullable=False, default=dt.datetime.utcnow)
        finished = Column(DateTime, nullable=True)
        done = Column(Boolean, nullable=False, default=False)

    class DataVisionModel(DataBase):
        __tablename__ = "data_vision"
        x = Column(DateTime, nullable=False, primary_key=True)
        id_item = Column(Integer, nullable=False, primary_key=True)
        yhat = Column(Integer, nullable=False)
        error = Column(Integer, nullable=True)

    class TempDataVisionModel(DataBase):
        __tablename__ = "tmp_vision"
        x = Column(DateTime, nullable=False, primary_key=True)
        id_item = Column(Integer, nullable=False, primary_key=True)
        yhat = Column(Integer, nullable=True)

    class AutotuneVisionModel(DataBase):
        __tablename__ = "autotune"
        datetime = Column(DateTime, nullable=False, primary_key=True)
        hparams = Column(Text, nullable=False, primary_key=True)
        training_time = Column(Integer, nullable=True)
        mean_error = Column(Integer, nullable=True)
        results = Column(Text, nullable=True)


    @staticmethod
    def connect(dbname='default'):
        # Create an SQLAlchemy engine with a connection pool
        db_url = 'sqlite:///instance/'+dbname+'.db'
        engine = create_engine(db_url, connect_args={'timeout': 2})

        if dbname == 'default':
            DataStorage.DefaultBase.metadata.create_all(engine)
        else:
            DataStorage.DataBase.metadata.create_all(engine)

        return engine

    @staticmethod
    def close(engine):
        engine.dispose()


class DataSource:

    @staticmethod
    def from_json(in_json):
        type = in_json['type']
        params = in_json['params']

        obj = None
        if type == 'datasklad':
            obj = Datasklad(params['id_item'])
        elif type == 'dummy':
            obj = None
        else:
            pass

        return obj

    def __init__(self):
        self.name = ""
        pass

    def __str__(self):
        return "DataSource Abstract"

    def get_data(self, ts_from, ts_to):
        # return pandas.DataFrame
        pass

#@TODO Refactor to SQLAlchemy before it's to late!!
class Datasklad(DataSource):

    def __init__(self, id_item):
        super().__init__()
        self.id_item = id_item
        engine = self.connect()

        # Execute a raw SQL query
        with engine.connect() as connection:
            result = connection.execute(text(f"""SELECT popis FROM konfig WHERE id_item = {self.id_item}"""))

            # Fetch the first row from the result
            row = result.fetchone()
            if row:
                # Assuming popis is the name column you're fetching
                self.name = row[0]
            else:
                # Handle the case where no rows are returned
                self.name = None

        # Don't forget to dispose of the engine when done
        self.close(engine)


        # Create a session factory
        # self.Session = sessionmaker(bind=self.db_engine)

    def __str__(self):
        return self.name


    def connect(self):
        # Create an SQLAlchemy engine with a connection pool
        db_url = 'postgresql://<username>:<password>@<hostname>:<port>/<database>'
        engine = create_engine(db_url)
        return engine

    def close(self, engine):
        engine.dispose()

    def get_data(self, ts_from, ts_to):

        # conn = engine
        conn = self.connect()

        # Define your SQL query
        # sql_query = "SELECT " \
        #             "date_trunc('hour', datum) AS x, " \
        #             "valuer AS y " \
        #             "FROM data_r " \
        #             "WHERE id_item = "+str(self.id_item)+" AND " \
        #             "datum >= '"+get_datetime_str(ts_from)+"' AND datum < '"+get_datetime_str(ts_to)+"' " \
        #             "ORDER BY datum ASC "

        sql_query = """
            SELECT
                cal AS x,
                d.y AS y
            FROM generate_series(date_trunc('hour', '"""+get_datetime_str(ts_from)+"""'::timestamp), 
                                 '"""+get_datetime_str(ts_to)+"""'::timestamp - '1 second'::interval, 
                                 '1 hour'::interval
             ) cal
            LEFT JOIN (

                SELECT
                    date_trunc('hour', datum) AS x,
                    valuer AS y
                FROM data_r
                WHERE id_item = """+str(self.id_item)+"""
                    AND datum >= '"""+get_datetime_str(ts_from)+"""' AND datum < '"""+get_datetime_str(ts_to)+"""'
            ) d ON (d.x = cal)

        """

        # print("DB SELECT")
        # print(sql_query)
        # Use pandas to read the query results into a DataFrame
        df = pd.read_sql_query(sql_query, conn)
        # print("DB DONE")

        # Close the database connection
        #conn.close()
        self.close(conn)

        return df


class Series:

    @staticmethod
    def from_json(input):
        datasource = DataSource.from_json(input)
        series = Series(datasource)
        return series
        pass

    def __init__(self, datasource):

        self.source = datasource
        self.df = False
        self.units = 'm3/h'

    def __str__(self):
        return str(self.source)


    def load_data(self, ts_from, ts_to, raw=False, analyse=False):

        df = self.source.get_data(ts_from, ts_to)

        df['x'] = pd.to_datetime(df['x'])
        df.set_index('x', inplace=True)
        df = df.asfreq('h')
        df.index.freq = 'h'

        if not raw:

            # Mask out zero values
            df['y'] = df['y'].replace(0, None)

            # Mask out outliers
            # - calculate the IQR (Interquartile Range)
            Q1 = df['y'].quantile(0.25)
            Q3 = df['y'].quantile(0.75)
            IQR = Q3 - Q1

            # - define the lower and upper bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # - find outliers
            df['y'] = df['y'].mask(
                (df['y'] < lower_bound) | (df['y'] > upper_bound))

            # Interpolate None values
            df['y'] = df['y'].interpolate(method='linear')
            # replace boundary None values
            df['y'] = df['y'].fillna(df['y'].mean())
            # Drop any leftover None values
            df = df.dropna()

            if analyse:
                df['y_mean'] = df['y'].mean()
                df['y_lo'] = lower_bound
                df['y_hi'] = upper_bound

        self.df = df
        return self.df

    def acf(self):
        from statsmodels.tsa.stattools import acf
        autocorr = acf(self.df['y'], nlags=30*24)

        return autocorr

    def adf(self):
        from statsmodels.tsa.stattools import adfuller
        adf_test = adfuller(self.df['y'])
        print(adf_test)

        return adf_test[0], adf_test[1]

    def decompose(self):
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomp = seasonal_decompose(self.df['y'], model='additive', period=24)

        return decomp

