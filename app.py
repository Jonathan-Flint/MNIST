import os
from dash import Dash, html, dcc, Input, Output, State, exceptions, no_update
import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import io
from PIL import Image

app = Dash(__name__)
W1 = pd.read_csv('Model Params/W1.csv')
W2 = pd.read_csv('Model Params/W2.csv')
b1 = pd.read_csv('Model Params/b1.csv')
b2 = pd.read_csv('Model Params/b2.csv')


def ReLu(x):
    return np.maximum(0, x)

def softmax(z):
    return np.exp(z) / sum(np.exp(z))
    

def prop_forwards(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def make_a_prediction(X, W1, b1, W2, b2):
    _, _, _, A2 = prop_forwards(W1, b1, W2, b2, X)
    preds = get_predictions(A2)
    return preds

def test_prediction(im, W1, b1, W2, b2):
    prediction = make_a_prediction(im, W1, b1, W2, b2)
    print("Prediction: ", prediction)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

app.layout = html.Div([
    dcc.Graph( id='graph', figure={ 'data': [], 'layout': { 'dragmode': 'drawopenpath'}}),
    html.Button('Save Image', id='btn-save-image'),
    html.Div(id='prediction'),
    html.Img(id='image-drawn')
])

@app.callback(
    Output('image-drawn', 'src'),
    Output('prediction', 'children'),
    Input('graph', 'relayoutData'),
    Input('btn-save-image', 'n_clicks'),
    State('graph', 'figure')
)
def save_image(relayoutData, n_clicks, fig):
    if n_clicks:
        src = 'https://www.python.org/static/community_logos/python-logo-master-v3-TM.png'
        return src, np.random.randint(1)
    src = 'https://www.python.org/static/community_logos/python-logo-master-v3-TM.png'
    return src, np.random.randint(1)

if __name__ == '__main__':
    if not os.path.exists('images'):
        os.makedirs('images')
    app.run_server(debug=True)