#Flask Application: Serves a web page with embedded Dash components.
#Dash Plotly Map: Displays real-time drone positions on a map using Scattermapbox.
#User Interaction: The interface allows users to view drone status and positions interactively.

from flask import Flask, render_template
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash_app = dash.Dash(__name__, server=app, routes_pathname_prefix='/dashboard/', external_stylesheets=external_stylesheets)

dash_app.layout = html.Div(children=[
    html.H1(children='Multi-Agent Autonomous Drones Dashboard'),

    dcc.Graph(
        id='drone-map',
        figure={
            'data': [
                go.Scattermapbox(
                    lat=['37.7749'],  # Example latitude
                    lon=['-122.4194'],  # Example longitude
                    mode='markers',
                    marker=dict(size=10),
                    text=['Drone 1']
                )
            ],
            'layout': go.Layout(
                autosize=True,
                hovermode='closest',
                mapbox=dict(
                    accesstoken='your_mapbox_access_token',
                    bearing=0,
                    center=dict(lat=37.7749, lon=-122.4194),
                    pitch=0,
                    zoom=10
                ),
            )
        }
    )
])

if __name__ == '__main__':
    app.run(debug=True)
