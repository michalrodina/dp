# Define a callback to update the line chart and filter the data
import dash

from dash import html, dcc, callback, Input, Output, State, ctx

import numpy as np
import pandas as pd
import vision
import visiondata as vd

@callback(
        [Output('series-chart', 'figure'),
         Output('series-content', 'children'),
         Output('series-persistent-dragmode', 'data')],
        [Input('series-chart', 'relayoutData')],
        State('series-persistent-dragmode', 'data')
)
def update_series_line_chart(relayout_data, persistence):
    print('Series', 'Persistence', persistence)
    print('Series', 'ReLeayoutData', relayout_data)
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

    if chart_type == 'series':
        p = vision.Predictor.db_load(id)
        series = p.series

        if not ts_to or ts_to == 'False':
            ts_to = p.ts_now
            ts_from = p.ts_history

        df_raw = series.load_data(ts_from, ts_to, raw=True).copy()
        df = series.load_data(ts_from, ts_to, analyse=True)

        print('Series', ts_from, ts_to)
        print(df)
        print(df['y'])

        acf = series.acf()
        decomp = series.decompose()
        adf_t, adf_p = series.adf()

        # Calculate statistical measures
        mean_val = np.mean(df['y'])
        median_val = np.median(df['y'])
        mode_val = df['y'].mode().values[0]  # Mode can have multiple values, so we take the first one
        std_dev = np.std(df['y'])
        min_val = np.min(df['y'])
        max_val = np.max(df['y'])
        range_val = max_val - min_val

        # Calculate the variance of each component
        trend_variance = np.var(decomp.trend)
        seasonal_variance = np.var(decomp.seasonal)
        residual_variance = np.var(decomp.resid)

        # Calculate the relative strength of trend and season
        # You can use variance ratio or any other measure of variability
        # For example, you can use the ratio of variances
        trend_strength = max(0, 1 - (residual_variance / (trend_variance + residual_variance)))
        seasonal_strength = max(0, 1 - (residual_variance / (seasonal_variance + residual_variance)))

        content.append(html.Div([
            html.H4("Základní statistické údaje"),
            html.P(f"Průměr: {round(mean_val, 3)} {series.units}"),
            html.P(f"Median: {round(median_val, 3)} {series.units}"),
            html.P(f"Modus: {round(mode_val, 3)} {series.units}"),
            html.P(f"Směrodatná odchylka: {round(std_dev, 3)} {series.units}"),
            html.P(f"Minimum: {round(min_val, 3)} {series.units}"),
            html.P(f"Maximum: {round(max_val, 3)} {series.units}"),

            html.H4("Dekompozice"),
            html.P(f"Síla trendové složky: {round(trend_strength, 3)}"),
            html.P(f"Síla sezónní složky: {round(seasonal_strength, 3)}"),
        ]))

        max_cor, max_cor_i = max((value, index + 2) for index, value in enumerate(acf[2:]))
        content.append(html.Div(f"Max. autokorelace {round(max_cor, 3)} při {max_cor_i} vzorcích"))

        # print(acf)
        acf_figure_data = [
            {'x': list(range(len(acf))), 'y': acf, 'name': 'Autocorrelation'},
        ]

        print(adf_t, adf_p)
        content.append(html.Div(f"ADF P-hodnota: {adf_p}"))

        # print(decomp)

        if not ts_to:

            ts_to = pd.to_datetime(str(df.index[-1]))
            ts_from = pd.to_datetime(str(df.index[0]))

        figure_data = [
            {'x': df_raw.index, 'y': df_raw['y'], 'name':  f"Zdrojová data [{series.units}]"},
            {'x': df.index, 'y': df['y'], 'name': f"Vyčištěná data [{series.units}]"},
            {'x': df.index, 'y': df['y_mean'], 'name': f"Průměr [{series.units}]"},
            # {'x': df.index, 'y': df['y_lo'], 'name': f"Dolní mez [{series.units}]"},
            {'x': df.index, 'y': df['y_hi'], 'name': f"Horní mez [{series.units}]"},

            # {'x': df.index, 'y': decomp.seasonal, 'name': 'Seasonal'},
            # {'x': df.index, 'y': decomp.trend, 'name': 'Trend'},
            # {'x': df.index, 'y': decomp.resid, 'name': 'Residual'},
            #                {'x': df.index, 'y': df['yhat'], 'name': 'Forecast'},
            #                {'x': df.index, 'y': df['error'], 'name': 'Error'},

        ]

        decomp_figure_data = [
            {'x': df_raw.index, 'y': df_raw['y'],
             'name':  f"Zdrojová data [{series.units}]",
             'type': 'scatter', 'mode': 'lines',
             'xaxis': 'x', 'yaxis': 'y1'},
            {'x': df.index, 'y': decomp.trend,
             'name': f"Trendová složka [{series.units}]",
             'type': 'scatter', 'mode': 'lines',
             'xaxis': 'x', 'yaxis': 'y2'},
            {'x': df.index, 'y': decomp.seasonal,
             'name': f"Sezónní složka [{series.units}]",
             'type': 'scatter', 'mode': 'lines',
             'xaxis': 'x', 'yaxis': 'y3'},

            {'x': df.index, 'y': decomp.resid,
             'name': f"Residuální složka [{series.units}]",
             'type': 'scatter', 'mode': 'lines',
             'xaxis': 'x', 'yaxis': 'y4'},
        ]

    figure = {
        'data': figure_data,
        'layout': {
            'dragmode': persistence['dragmode'],
            'legend': {'orientation': 'h',},
            'xaxis': {'title': ''},
            'yaxis': {'title': f"Hodnota veličiny [{series.units}]"}
        }
    }

    acf_figure = {
        'data': acf_figure_data,
        'layout': {
            'legend': {'orientation': 'h', 'y': -0.2},
            'xaxis': { 'title': 'Zpoždění [vzorky]'},
            'yaxis': {'title': 'Míra autokorelace [-]'}
        }

    }

    decomp_figure = {
        'data': decomp_figure_data,
        'layout': {
            'legend': {'orientation': 'h', 'y': -0.2},
            "xaxis": {"title": "Čas", "domain": [0, 1], "ticktext": '', "anchor": "y4"},
            "yaxis1": {"title": f"Hodnota [{series.units}]", "domain": [0.75, 1]},
            "yaxis2": {"title": f"T [{series.units}]", "domain": [0.50, 0.74]},
            "yaxis3": {"title": f"S [{series.units}]", "domain": [0.25, 0.49]},
            "yaxis4": {"title": f"R [{series.units}]", "domain": [0, 0.24]},
        }

    }

    content.append(dcc.Graph(id='series-acf-chart',
                             style={
                                 'width': '100%',
                                 'height': '400px'
                             },
                             figure=acf_figure,
                             config={'staticPlot': False}))

    content.append(dcc.Graph(id='series-decomp-chart',
                             style={
                                 'width': '100%',
                                 'height': '800px'
                             },
                             figure=decomp_figure,
                             config={'staticPlot': False}))

    return figure, content, {'chart': chart_type, 'db_id': id, 'xaxis.range[0]': ts_from, 'xaxis.range[1]': ts_to, 'dragmode': dragmode}

def layout(id=None, **other_unknown_query_strings):
    if not id:
        #preds = vd.DataStorage.PredictorModel.query.all()

        layout = html.Div([
            html.H1('Database Table'),
            # dcc.Table(
            #     id='database-table',
            #     columns=[{'name': col, 'id': col} for col in preds.columns],
            #     data=preds.to_dict('records')
            # )
        ])

        return layout
    else:
        print('Dash series show()', id)
        layout = html.Div([
            html.H1(f"Predictor show {id}"),
            dcc.Graph(id='series-chart',
                      style={
                          'width': '100%',
                          'height': '400px'
                      },
                      config={'staticPlot': False}),
            html.Div(id='series-content'),
            dcc.Store(id='series-persistent-dragmode', data={'chart': 'series',
                                                      'db_id': id,
                                                      'dragmode': 'zoom',
                                                      'xaxis.range[0]': False,
                                                      'xaxis.range[1]': False})
        ])

    return layout


dash.register_page(__name__, path_template="/series/show/<id>", layout=layout)