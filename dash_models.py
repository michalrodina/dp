import dash
from dash import html, dcc, callback, Input, Output
import visiondata as vd
import vision
from sqlalchemy import text


def layout():
    preds = [
        {'name': 'SARIMA', 'db_id': 8},
        {'name': 'LSTM', 'db_id': 10},
        {'name': 'Prophet', 'db_id': 9},
    ]


    vx = []
    for p in preds:
        values = []
        pr = vision.Predictor.db_load(p['db_id'])

        results = pr.get_autotune_results(num_rows=0)
        engine = pr.storage_open()
        # Execute a raw SQL query
        with engine.connect() as connection:
            result = connection.execute(text("""SELECT hparams, COUNT(*), AVG(training_time), 
                        MIN(mean_error), AVG(mean_error), MAX(mean_error)
                        FROM autotune 
                        WHERE datetime > '2024-03-01 00:00:00'
                        GROUP BY hparams
                        -- HAVING json_extract(hparams, '$.order') LIKE '%1,0%'
                        
                        ORDER BY AVG(mean_error) ASC """))

            # Fetch the results, if any
            for r in result:
                print(r)

                values.append([r[4], r[2], f"{p['name']}{r.hparams}", p['name'], p['db_id']])
            if len(values) > 0:
                vx.append(values)

        # Don't forget to dispose of the engine when done
        engine.dispose()

        # for r in results:
        #     values.append([r.mean_error, r.training_time, f"{p['name']}{r.hparams}", p['name'], p['db_id']])
        # if len(values) > 0:
        #     vx.append(values)
    # pred = vision.Predictor.db_load(id)
    # results = pred.get_autotune_results(num_rows=15)
    #
    # values = [[r.mean_error, r.training_time, f"{r.hparams}"] for r in results]
    # # text = [f"{r.hparams}  {r.training_time} {r.results}" for r in results]

    layout = html.Div([
        html.H1(f"Models autotune results"),
        html.Div(id='content', children=[
            # dash.dash_table.DataTable(sorted([v for values in vx for v in values], key=lambda x:x[0]), [{'name': c, 'id': i} for i, c in
            #                                    enumerate(['Mean Error', 'Training Time', 'Configuration'])])

        ]),
        # dcc.Graph(
        #     id='error-bar-chart',
        #     figure={
        #         'data': [
        #             {
        #                 'x': [v[0] for v in sorted(values, key=lambda x: x[0])],
        #                 'y': list(range(len(values))),
        #                 'text': [f"{v[2]}\n Doba tréninku: {v[1]}" for v in sorted(values, key=lambda x: x[0])],
        #                 'type': 'bar',
        #                 'orientation': 'h',
        #                 'hoverinfo': 'x',
        #                 'name': values[0][3]
        #             } for values in vx
        #         ],
        #         'layout': {
        #             'title': 'Srovnání modelů dle přesnosti',
        #             'xaxis': {'title': 'Průměrná chyba predikce'},
        #             'yaxis': {'title': False, 'showticklabels': False},  # Hide y-axis tick labels
        #             'margin': {'l': 100, 'r': 100, 't': 50, 'b': 50}
        #         }
        #     },
        #     style={'height': '500px'}
        # ),
        # dcc.Graph(
        #     id='time-bar-chart',
        #     figure={
        #         'data': [
        #             {
        #                 'x': [v[1] for v in sorted(values, key=lambda x: x[1])],
        #                 'y': list(range(len(values))),
        #                 'text': [f"{v[2]}, přesnost: {v[0]}" for v in sorted(values, key=lambda x: x[1])],
        #                 'type': 'bar',
        #                 'orientation': 'h',
        #                 'hoverinfo': 'x',
        #                 'name': values[0][3]
        #             } for values in vx
        #         ],
        #         'layout': {
        #             'title': 'Srovnání nejlepších modelů dle časové náročnosti',
        #             'xaxis': {'title': 'Doba tréninku [s]'},
        #             'yaxis': {'title': False, 'showticklabels': False},  # Hide y-axis tick labels
        #             'margin': {'l': 100, 'r': 100, 't': 50, 'b': 50}
        #         }
        #     },
        #     style={'height': '500px'}
        # ),
        dcc.Graph(
            id='model-scatter-chart',
            figure={
                'data': [
                    {
                        'x': [v[0] for v in values],
                        'y': [v[1] for v in values],
                        'text': [f"{v[2]} <br>Průměrná chyba: {round(v[0], 3)} <br>Čas tréninku: {round(v[1], 2)} s" for
                                 v in values],
                        'type': 'scatter',
                        'mode': 'markers',
                        'hoverinfo': 'text',
                        'name': values[0][3]
                    } for values in vx
                ],
                'layout': {
                    'title': 'Srovnání nejlepších modelů',
                    'xaxis': {'title': f"Průměrná chyba [{pr.series.units}]"},
                    'yaxis': {'title': f"Doba tréninku [s]"},  # Hide y-axis tick labels
                    'margin': {'l': 100, 'r': 100, 't': 50, 'b': 50},
                    # 'annotations': [
                    #     {
                    #         'x': x,
                    #         'y': y,
                    #         'text': text,
                    #         'showarrow': False,
                    #         'xanchor': 'center',  # Position text box at marker center
                    #         'yanchor': 'bottom',  # Position text box below marker
                    #         'font': {'color': 'black', 'size': 10}  # Adjust text font
                    #     }
                    #     for x, y, text in values
                    # ]
                }
            },
            style={'height': '500px'}
        )
    ])

    return layout


dash.register_page(__name__, path="/predictor/models", layout=layout)

# @callback(
#     Output('analytics-output', 'children'),
#     Input('analytics-input', 'value')
# )
# def update_city_selected(input_value):
#     return f'You selected: {input_value}'