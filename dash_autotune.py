import dash
from dash import html, dcc, callback, Input, Output
import visiondata as vd
import vision
from sqlalchemy import text

def layout(id=None):

    pred = vision.Predictor.db_load(id)
    results = pred.get_autotune_results(num_rows=15)

    values = [[r.mean_error, r.training_time, f"{r.hparams}"] for r in results]
    # text = [f"{r.hparams}  {r.training_time} {r.results}" for r in results]

    values = []
    engine = pred.storage_open()
    # Execute a raw SQL query
    with engine.connect() as connection:
        result = connection.execute(text("""SELECT hparams, COUNT(*), AVG(training_time), 
                            MIN(mean_error), AVG(mean_error), MAX(mean_error)
                            FROM autotune 
                            WHERE datetime > '2024-03-01 00:00:00'
                            GROUP BY hparams
                            -- HAVING json_extract(hparams, '$.order') LIKE '%1,0%'
                            HAVING COUNT(*) > 1
                            ORDER BY AVG(mean_error) ASC """))

        # Fetch the results, if any
        for r in result:
            print(r)

            values.append([r[4], r[2], f"{pred.name}{r.hparams}", pred.name, id])



    layout = html.Div([
        html.H1(f"Predictor #{id} autotune results"),
        html.Div(id='content', children=[
           dash.dash_table.DataTable(values, [{'name': c, 'id': i} for i, c in enumerate(['Mean Error', 'Training Time', 'Configuration'])])

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
        #                 'hoverinfo': 'x'
        #           }
        #       ],
        #         'layout': {
        #             'title': 'Srovnání modelů dle přesnosti',
        #             'xaxis': {'title': f"Průměrná chyba predikce [{pred.series.units}]"},
        #             'yaxis': {'title': False, 'showticklabels': False},  # Hide y-axis tick labels
        #             'margin': {'l': 100, 'r': 100, 't': 50, 'b': 50}
        #         }
        #   },
        # style={'height': '500px'}
        # ),
        dcc.Graph(
            id='time-bar-chart',
            figure={
                'data': [
                    {
                        'x': [v[1] for v in sorted(values, key=lambda x: x[1])],
                        'y': list(range(len(values))),
                        'text': [f"{v[2]}, přesnost: {v[0]} [{pred.series.units}]" for v in sorted(values, key=lambda x: x[1])],
                        'type': 'bar',
                        'orientation': 'h',
                        'hoverinfo': 'x'
                    }
                ],
                'layout': {
                    'title': 'Srovnání nejlepších modelů dle časové náročnosti',
                    'xaxis': {'title': 'Doba tréninku [s]'},
                    'yaxis': {'title': False, 'showticklabels': False},  # Hide y-axis tick labels
                    'margin': {'l': 100, 'r': 100, 't': 50, 'b': 50}
                }
            },
        style={'height': '500px'}
        ),
        dcc.Graph(
            id='model-scatter-chart',
            figure={
                'data': [
                    {
                        'x': [v[0] for v in values],
                        'y': [v[1] for v in values],
                        'text': [f"{v[2]} <br>Průměrná chyba: {round(v[0], 3)} [{pred.series.units}] <br>Čas tréninku: {round(v[1], 2)} [s]" for v in values],
                        'type': 'scatter',
                        'mode': 'markers',
                        'hoverinfo': 'text'
                    }
                ],
                'layout': {
                    'title': 'Srovnání nejlepších modelů',
                    'xaxis': {'title': f"Průměrná chyba [{pred.series.units}]"},
                    'yaxis': {'title': f"Doba tréninku [s]"},  # Hide y-axis tick labels
                    'margin': {'l': 100, 'r': 100, 't': 50, 'b': 50},
                }
            },
        style={'height': '500px'}
        )
    ])

    return layout

dash.register_page(__name__, path_template="/predictor/autotune/<id>", layout=layout)
