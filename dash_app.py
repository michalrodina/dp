# my_dash_app.py
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from flask import request

import plotly.express as px

import pandas as pd
import datetime as dt
import visiondata as vd
import vision

def build_dash(server):
    app = dash.Dash(__name__,
                    server=server,
                    url_base_pathname='/dash/',
                    use_pages=True,
                    pages_folder="",
                    external_stylesheets=[dbc.themes.BOOTSTRAP])
    import dash_autotune
    import dash_predictor_show
    import dash_predictors_show
    import dash_series_show
    import dash_test
    import dash_models
    import dash_manual

    app.layout = html.Div([
        dbc.Container([
            dbc.Row([
                html.H1("Datasklad | Vision"),
            ]),
            dbc.Row([
                html.Div([
                    html.Div(
                        dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
                    ) for page in dash.page_registry.values() if not page.get("params")
                ]),
            ]),
            dbc.Row([
                dash.page_container
            ]),
            dbc.Row([
                html.Div([
                    html.P(["Univerzita Jana Evangelisty Purkyně v Ústí nad Labem; Vodárny a kanalizace Karlovy Vary, a.s. | 2024"])
                ])
            ])

        ])
    ])

if __name__ == "__main__":
    app = build_dash(False)
    app.run_server(debug=True)
