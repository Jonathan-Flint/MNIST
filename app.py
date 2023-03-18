from dash import Dash, html, dcc, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
import matplotlib.pyplot as plt

app = Dash(__name__)
W1 = pd.read_csv('Model Params/W1.csv')
W2 = pd.read_csv('Model Params/W2.csv')
b1 = pd.read_csv('Model Params/b1.csv')
b2 = pd.read_csv('Model Params/b2.csv')
print(W1.shape, W2.shape, b1.shape, b2.shape)


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