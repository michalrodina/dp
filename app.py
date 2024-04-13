# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

from dash_app import build_dash

from vision import Predictor
import datetime
import uuid
import json

def pipe_query(pipe, query):
    pipe.send(query)
    result = False
    if pipe.poll(timeout=1):
        result = pipe.recv()
    return result


def build_app(pipe=False):
    if not pipe:
        print('Warning', 'No data pipe between modules!')

    # Initialize the Flask application
    app = Flask(__name__,
                template_folder="tpl")
    app.app_context().push()

    # Import Dash application
    dash = build_dash(app)

    # Configure the SQLite database
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///default.db'
    db = SQLAlchemy(app)

    # Define the model for the data (e.g., a "PredictorModel" model)
    class PredictorModel(db.Model):
        __tablename__ = "predictor"
        id = db.Column(db.Integer, primary_key=True)
        guid = db.Column(db.String(36), nullable=False)
        title = db.Column(db.String(200), nullable=False)
        datasource = db.Column(db.Text, nullable=False)
        model = db.Column(db.Text, nullable=False)
        ts_now = db.Column(db.DateTime, nullable=False)

    # Create the database tables
    db.create_all()

    # Define routes for CRUD operations
    @app.route('/')
    def index():

        preds = PredictorModel.query.all()

        result = pipe_query(pipe, 'test')

        print('GUI', 'Engine', result)
        return render_template('index.html', preds=preds)

    @app.route('/ajax/predictor/form', methods=['POST'])
    def ajax_predictor_form():
        print(request.data.decode('utf-8'))
        request_data = json.loads(request.data.decode('utf-8'))
        pred_type = request_data['model']
        pred_id = request_data['id']
        print(pred_type, pred_id)
        try:
            pred_id = int(pred_id)
        except:
            pred_id = 0

        if pred_id > 0:
            pred = PredictorModel.query.get(pred_id)
            pred.model = json.loads(pred.model)
            pred.datasource = json.loads(pred.datasource)
        else:
            pred=False
        return render_template('predictor.form.ajax.html', pred_type=pred_type, pred=pred)
        pass

    @app.route('/create', methods=['GET', 'POST'])
    def create():
        pred = PredictorModel()
        if request.method == 'POST':
            title = request.form.get('title')
            pred = PredictorModel(title=title, guid=str(uuid.uuid4()))

            ds_type = request.form.get('ds_type')
            ds_id_item = request.form.get('ds_id_item')
            datasource = {'type': ds_type,
                          'params': {'id_item': ds_id_item}}
            pred.datasource = json.dumps(datasource)
            pred_type = request.form.get('pred_type')
            model = {}
            print(pred_type)
            if pred_type == 'sarimax':
                p = int(request.form.get('pred_param[p]'))
                d = int(request.form.get('pred_param[d]'))
                q = int(request.form.get('pred_param[q]'))

                sp = int(request.form.get('pred_param[sp]'))
                sd = int(request.form.get('pred_param[sd]'))
                sq = int(request.form.get('pred_param[sq]'))

                sn = int(request.form.get('pred_param[sn]'))

                model = {'type': 'sarimax',
                         'params': {
                             'order': [p, d, q],
                             'sorder': [sp, sd, sq, sn]
                         }
                     }

            if pred_type == 'lstm':
                n_layers = int(request.form.get('pred_param[n_layers]'))
                n_neurons = int(request.form.get('pred_param[n_neurons]'))
                n_epochs = int(request.form.get('pred_param[n_epochs]'))


                model = {'type': 'sarimax',
                         'params': {
                             'n_layers': n_layers,
                             'n_neurons': n_neurons,
                             'n_epochs': n_epochs,

                         }
                     }

            pred.model = json.dumps(model)

            pred.ts_now = datetime.datetime.strptime(request.form.get('ts_now'), '%Y-%m-%d %H:%M:%S')
            db.session.add(pred)
            db.session.commit()
            return redirect('/')
        return render_template('update.html', pred=pred)

    @app.route('/update/<int:db_id>', methods=['GET', 'POST'])
    def update(db_id):
        pred = PredictorModel.query.get(db_id)
        pred.datasource = json.loads(pred.datasource)
        pred.model = json.loads(pred.model)
        if request.method == 'POST':
            pred.title = request.form.get('title')
            ds_type = request.form.get('ds_type')
            ds_id_item = request.form.get('ds_id_item')
            datasource = {'type': ds_type, 'params': {'id_item': ds_id_item}}

            model = {'type': request.form.get('pred_type')}

            if model['type'] == 'sarimax':
                model['params'] = {
                    'order': [
                        request.form.get('pred_param[p]'),
                        request.form.get('pred_param[d]'),
                        request.form.get('pred_param[q]')
                    ],
                    'sorder': [
                        request.form.get('pred_param[sp]'),
                        request.form.get('pred_param[sd]'),
                        request.form.get('pred_param[sq]'),
                        request.form.get('pred_param[sn]'),
                    ]
                }

            pred.model = json.dumps(model)
            pred.datasource = json.dumps(datasource)
            pred.ts_now = datetime.datetime.strptime(request.form.get('ts_now'), '%Y-%m-%d %H:%M:%S')
            db.session.commit()
            return redirect('/')
        return render_template('update.html', pred=pred)

    @app.route('/delete/<int:db_id>')
    def delete(db_id):
        task = PredictorModel.query.get(db_id)
        db.session.delete(task)
        db.session.commit()
        return redirect('/')

    return app

