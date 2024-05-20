# Define a callback to update the line chart and filter the data
import dash
import dash_table

from dash import html, dcc, callback, Input, Output, State, ctx

import pandas as pd
import vision
import visiondata as vd
from sqlalchemy import text

def layout():

    layout = html.Div([
        html.H1(f"Predictors showcase "),
        html.Div(id='predictors-content'),
        dcc.Graph(id='predictors-line-chart',
                  style={
                      'width': '100%',
                      'height': '800px'
                  },
                  config={'staticPlot': False}
                  ),
        dcc.Graph(id='predictors-box-plot',
                  style={
                      'width': '100%',
                      'height': '800px'
                  },
                  config={'staticPlot': False}
                  ),
        dcc.Graph(id='times-box-plot',
                  style={
                      'width': '100%',
                      'height': '800px'
                  },
                  config={'staticPlot': False}
                  ),

        dcc.Store(id='predictors-persistent-dragmode', data={'chart': 'predictor',
                                                  # 'ids': [16, 17, 23, 24, 14, 15, 18, 25, 26, 19],
                                                  # 'ids': [23, 24, 10, 14, 19],
                                                  'ids': [23, 24, 25, 26, 19], # pripadovka
                                                  # 'ids': [21, 27, 29, 31], #, 33], # testování
                                                  'dragmode': 'zoom',
                                                  'xaxis.range[0]': False,
                                                  'xaxis.range[1]': False})
    ])

    return layout

dash.register_page(__name__, layout=layout, path="/predictors/show")

@callback(
    [Output('predictors-line-chart', 'figure'),
     Output('predictors-box-plot', 'figure'),
     Output('times-box-plot', 'figure'),
     Output('predictors-content', 'children'),
     Output('predictors-persistent-dragmode', 'data')],
    [Input('predictors-line-chart', 'relayoutData')],
    State('predictors-persistent-dragmode', 'data')
)
def update_predictors_line_chart(relayout_data, persistence):
    print('Predictor', 'Persistence', persistence)
    print('Predictor', 'ReLeayoutData', relayout_data)
    print(ctx.triggered_id, ctx.inputs, ctx.states)
    chart_type = persistence['chart']

    content = []
    # if this is update callback
    if relayout_data is not None:
        # persistent dragmode
        dragmode = relayout_data.get('dragmode', persistence['dragmode'])
        persistence['dragmode'] = dragmode

        # if axis range is to be automatic reset it
        if 'xaxis.autorange' in relayout_data:
            ts_from = False
            ts_to = False

        # setup x axis range
        else:
            ts_from = str(relayout_data.get('xaxis.range[0]', persistence['xaxis.range[0]']))[:19].replace('T', ' ')
            ts_to = str(relayout_data.get('xaxis.range[1]', persistence['xaxis.range[1]']))[:19].replace('T', ' ')

        # persist the xaxis
        persistence['xaxis.range[0]'] = ts_from
        persistence['xaxis.range[1]'] = ts_to

    # initial setup
    else:
        dragmode = persistence['dragmode']
        ts_from = False
        ts_to = False

    figure_data = []
    box_data = []
    times_data = []
    preds_data = []
    print(ts_from, ts_to)

    if chart_type == 'predictor':
        ids = persistence['ids']
        figure_data = []
        for id in ids:
            p = vision.Predictor.db_load(id)
            print('PRedictors ', ts_from, ts_to)
            print(p.name)
            d = vision.Detector(p, ts_from=ts_from, ts_to=ts_to)

            df = d.analyze()
            df = df.copy()
            print(df.index)
            # if not ts_to:
            #     ts_to = pd.to_datetime(str(df.index[-1]))
            #     ts_from = pd.to_datetime(str(df.index[0]))



            print(df.columns)



            if id == ids[0]:
                figure_data.append({'x': df.index, 'y': df['y'], 'name': f"Skutečnost: y [{p.series.units}]"})
            figure_data.append({'x': df.index, 'y': df['yhat'], 'name': f"{p} Predikce: y^ [{p.series.units}]"})

            box_data.append({'type': 'box',

                             'x': f"{p.guid}",
                             # 'y': df['error']/df['y_mean'],
                             # 'boxpoints': 'outliers',
                             'y': df['error'],
                             'boxpoints': False,
                             'name': f"#{id}: {p} - {p.series}"})

            engine = vd.DataStorage.connect()
            # Execute a raw SQL query
            with engine.connect() as connection:
                result = connection.execute(text(f"""
                    SELECT * FROM (
                        SELECT 
                            guid, 
                            (julianday(finished) - julianday(started)) * 24.0 * 60.0 * 60.0 AS elapsed 
                        FROM job 
                        WHERE guid = '{p.guid}'
                    ) WHERE elapsed > 0.05
                """))

                times_data.append({'type': 'box',
                                   'x': f"{p.guid}",
                                   'y': [r[1] for r in result],
                                   'name': f"#{id}: {p} - {p.series}"})

            # Don't forget to dispose of the engine when done
            vd.DataStorage.close(engine)

            # figure_data.append({'x': df.index, 'y': df['error_mean'], 'name': f"P{id}"+'Chyba: sqrt(y - y^)^2'})
            # figure_data.append({'x': df.index, 'y': df['diff_mean'], 'name': f"P{id}"+'Rozdíl: y - y^'})
            # figure_data.append({'x': df.index, 'y': df['diff_mean']+2.0*df['diff_std'], 'name': 'Chyba: hranice'})

            # figure_data.append({'x': df.index, 'y': df['diff_mean_8'], 'name': f"P{id}"+'Chyba: klouzavý pr. (8h)'})
            # figure_data.append({'x': df.index, 'y': df['diff_mean_24'], 'name': f"P{id}"+'Chyba: klouzavý pr. (24h)'})
            #
            # figure_data.append({'x': df.index, 'y': df['diff_cumm_8'], 'name': f"P{id}"+'Chyba: kumulativní (8h)'})
            # figure_data.append({'x': df.index, 'y': df['diff_cumm_24'], 'name':  f"{p} "+'Chyba: kumulativní (24h)'})

            preds_data.append({
                'id': id,
                'error_mean': df['error_mean'][0],
                'error_std': df['error_std'][0],
                'diff_mean': df['diff_mean'][0],
                'diff_std': df['diff_std'][0],
                'series_mean': p.series.df['y'].mean(),
                'series_std': p.series.df['y'].std()
            })

        content.append(dash_table.DataTable(preds_data))

    figure = {
        'data': figure_data,
        'layout': {
            'dragmode': persistence['dragmode'],
            'xaxis': {'title': f"Čas"},
            'yaxis': {'title': f"Hodnota veličiny [{p.series.units}]"},
            'legend': {'orientation': 'h', 'y': -0.15}
        }}

    print(box_data)

    # Create box plot JSON
    figure_box = {
        'data': box_data,
        'layout': {
            'dragmode': persistence['dragmode'],
            'xaxis': {'title': 'Prediktor', 'showticklabels': False},
            'yaxis': {'title': f"Chyba přepovědí [{p.series.units}]"},
            # 'yaxis': {'title': f"Normalizovaná chyba přepovědí"},
            'legend': {'orientation': 'h',  'y': -0.15}
        }}

    # Create box plot JSON
    figure_times = {
        'data': times_data,
        'layout': {
            'dragmode': persistence['dragmode'],
            'xaxis': {'showticklabels': False, 'title': 'Prediktor'},
            'yaxis': {'title': 'Doba tréninku [s]'},
            'legend': {'orientation': 'h', 'y': -0.15}
        }}

    return figure, figure_box, figure_times, content, {'chart': chart_type, 'ids': ids, 'xaxis.range[0]': ts_from, 'xaxis.range[1]': ts_to,
                             'dragmode': dragmode}