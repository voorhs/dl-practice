# taken from https://dash.plotly.com/dash-core-components/tooltip?_gl=1*dzxk1a*_ga*MTEwMTg2ODY3OC4xNjg2NzY0MDE0*_ga_6G7EE0JNSC*MTY4Njc2NDAxMy4xLjEuMTY4Njc2NDQyMi4wLjAuMA..#visualizing-t-sne-plot-of-mnist-images

import io
import base64

from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go

from PIL import Image

from sklearn.manifold import TSNE
import numpy as np


# helper function
def np_image_to_base64(im_matrix):
    im = Image.fromarray(np.moveaxis(255 * im_matrix, 0, 2).astype(np.uint8))
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

# main function
def visualize(embeddings, images, labels, targets) -> JupyterDash:
    """
    Project images vector representations to plane and visualize
    with original images and labels appearing on hover
    """

    # project embeddings to plane
    tsne = TSNE(n_components=2, random_state=0).fit_transform(embeddings)

    # generate color (hex code) for each class label
    n_labels = len(np.unique(labels))
    hex_syms = [a for a in '0123456789abcdef']
    base_colors = ['#' + ''.join(a for a in np.random.choice(hex_syms, size=6)) for i in range(n_labels)]
    colors = [base_colors[i] for i in targets]

    # scatter plot (so far without images appearing on hover)
    fig = go.Figure(data=[go.Scatter(
        x=tsne[:, 0],
        y=tsne[:, 1],
        mode='markers',
        marker={'size': 4, 'color': colors},
        
    )])

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    # backend object
    app = JupyterDash(__name__)

    # html markup
    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ],
    )

    # add functionality
    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        # if any event
        if hoverData is None:
            return False, no_update, no_update

        # read event data
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        # encode image and send to db
        im_matrix = images[num]
        im_url = np_image_to_base64(im_matrix)
        
        # define popup info
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "100px", 'height': "100px",
                        'display': 'block', 'margin': '0 auto',
                        'image-rendering': 'pixelated'},
                ),
                html.P(str(labels[num]), style={'font-weight': 'bold'})
            ])
        ]

        return True, bbox, children

    return app