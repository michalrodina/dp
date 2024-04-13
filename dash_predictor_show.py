# Define a callback to update the line chart and filter the data
import dash

from dash import html, dcc, callback, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc

import pandas as pd
import vision

def layout(id=None):

    print('Dash predictor show()', id)
    layout = html.Div([
        dbc.Container([
            dbc.Row([
                html.H1(f"Predictor show {id}")
            ]),
            dbc.Row([
                html.Div(id='content'),
            ]),
            dbc.Row([
                dcc.Graph(id='line-chart',
                          style={
                              'width': '100%',
                              'height': '800px'
                          },
                          config={'staticPlot': False}),
            ]),
            dcc.Store(id='persistent-dragmode',
                      data={'chart': 'predictor',
                            'db_id': id,
                            'dragmode': 'zoom',
                            'xaxis.range[0]': False,
                            'xaxis.range[1]': False})
        ])
    ])

    return layout

dash.register_page(__name__, layout=layout, path_template="/predictor/show/<id>")

@callback(
    [Output('line-chart', 'figure'),
     Output('content', 'children'),
     Output('persistent-dragmode', 'data')],
    [Input('line-chart', 'relayoutData')],
    State('persistent-dragmode', 'data')
)
def update_predictor_line_chart(relayout_data, persistence):
    print('Predictor', 'Persistence', persistence)
    print('Predictor', 'ReLeayoutData', relayout_data)
    print(ctx.triggered_id, ctx.inputs, ctx.states)
    id = persistence['db_id']
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

    print(ts_from, ts_to)

    if chart_type == 'predictor':
        p = vision.Predictor.db_load(id)
        d = vision.Detector(p, ts_from=ts_from, ts_to=ts_to)

        df = d.analyze()
        if not ts_to:
            ts_to = pd.to_datetime(str(df.index[-1]))
            ts_from = pd.to_datetime(str(df.index[0]))

        figure_data = []

        figure_data.append({'x': df.index, 'y': df['y'], 'name': f"Skutečnost: y [{p.series.units}]"})
        figure_data.append({'x': df.index, 'y': df['yhat'], 'name': f"Predikce: y^ [{p.series.units}]"})

        # figure_data.append({'x': df.index, 'y': df['diff'], 'name': f"Rozdíl: y - y^ [{p.series.units}]"})
        # figure_data.append({'x': df.index, 'y': df['diff_mean'], 'name': f"Rozdíl: průměr [{p.series.units}]"})
        # # figure_data.append({'x': df.index, 'y': df['diff_mean']+2.0*df['diff_std'], 'name': 'Chyba: hranice'})
        #
        # # figure_data.append({'x': df.index, 'y': df['diff_cumm_8'], 'name': 'Rozdíl: kumulativní (8h)'})
        figure_data.append({'x': df.index, 'y': df['diff_cumm_24'], 'name': f"Rozdíl: kumulativní (24h) [{p.series.units}]"})
        # # figure_data.append( {'x': df.index, 'y': df['diff_cumm_24'], 'name': f"Chybovost predikce [{p.series.units}]"})
        # # figure_data.append({'x': df.index, 'y': df['y_mean_24'] + df['y_std_24'], 'name': f"Rozdíl: hranice (24h) [{p.series.units}]"})
        #
        # # figure_data.append({'x': df.index, 'y': df['error'], 'name': f"Chyba [{p.series.units}]"})
        # # figure_data.append({'x': df.index, 'y': df['error_mean'], 'name': f"Chyba: průměr [{p.series.units}]"})
        figure_data.append({'x': df.index, 'y': df['alarm_level'], 'name': f"Rozdíl: hranice (24h) [{p.series.units}]"})
        # figure_data.append({'x': df.index, 'y': df['alarm']*150.0, 'name': 'Rozdíl: mimo meze'})
        # figure_data.append({'x': df.index, 'y': df['diff_cumm_24_rel'], 'name': 'Rozdíl: poměrný (24h)'})
        #
        # figure_data.append({'x': df.index, 'y': df['error_mean'], 'name': 'Chyba: průměrná'})
        # figure_data.append({'x': df.index, 'y': df['error_mean_8'] / df['error_mean'], 'name': 'Chyba: průměrná (24h)'})

        # content.append(dash_table.DataTable({'mean'}))

    # if chart_type == 'series':
    #     p = vision.Predictor.db_load(id)
    #     series = p.series
    #
    #     # series = vd.Series(ds)
    #
    #     if not ts_to or ts_to == 'False':
    #         ts_to = p.ts_now
    #         ts_from = p.ts_history
    #
    #     df = series.load_data(ts_from, ts_to)
    #     decomp = series.decompose()
    #     acf = series.acf()
    #     max_cor, max_cor_i = max((value, index + 1) for index, value in enumerate(acf[1:]))
    #
    #     content.append(html.Div(str(max_cor) + ' at ' + str(max_cor_i) + ' lags'))
    #
    #     print(decomp)
    #
    #     if not ts_to:
    #         ts_to = pd.to_datetime(str(df.index[-1]))
    #         ts_from = pd.to_datetime(str(df.index[0]))
    #
    #     figure_data = [
    #         {'x': df.index, 'y': df['y'], 'name': 'Actual'},
    #         {'x': df.index, 'y': decomp.seasonal, 'name': 'Seasonal'},
    #         {'x': df.index, 'y': decomp.trend, 'name': 'Trend'},
    #         {'x': df.index, 'y': decomp.resid, 'name': 'Residual'},
    #         #                {'x': df.index, 'y': df['yhat'], 'name': 'Forecast'},
    #         #                {'x': df.index, 'y': df['error'], 'name': 'Error'},
    #
    #     ]

    figure = {
        'data': figure_data,
        'layout': {
            'dragmode': persistence['dragmode'],
            'legend': {'orientation': 'h', 'y': -0.15},
            "xaxis": {"title": "Čas"},
            "yaxis": {"title": f"Hodnota veličiny [{p.series.units}]"}
        }}

    return figure, content, {'chart': chart_type, 'db_id': id, 'xaxis.range[0]': ts_from, 'xaxis.range[1]': ts_to,
                             'dragmode': dragmode}