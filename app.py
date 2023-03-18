from dash import Dash, html, dcc, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go

app = Dash(__name__)
W1 = pd.read_csv('Model Params/W1.csv')
W2 = pd.read_csv('Model Params/W2.csv')
b1 = pd.read_csv('Model Params/b1.csv')
b2 = pd.read_csv('Model Params/b2.csv')
print(W1.shape, W2.shape, b1.shape, b2.shape)

app.layout = html.Div([
    html.H1('Digit Recognition'),
    html.Div([
        '''A Dash Web Application Using a Neural Network to Recognise Digits'''
    ]),
    dcc.Graph
    (
        id='input_canvas',
        figure=
        {
            'data':[],
            'layout':{
                'dragmode': 'drawopenpath'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)