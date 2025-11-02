import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State
from plotly.subplots import make_subplots

"""The following section is from the original source"""

# DATA LOADING SECTION (from original)
csv_file = "boardgame-geek-dataset_organized.csv"
json_file = "boardgamegeek.json"

# Read the organized CSV dataset
df = pd.read_csv(csv_file, encoding="utf-8")

# Read the JSON dataset if needed
df_json = pd.read_json(json_file)

print("CSV and JSON files loaded successfully.")

# CLEANING DATA
# For predictor building and most analyses, drop rows with missing critical values
cols_to_check = [
    "min_players",
    "max_players",
    "min_playtime",
    "max_playtime",
    "minimum_age",
    "avg_rating",
    "num_ratings",
    "complexity",
]
df_clean = df.dropna(subset=cols_to_check)

"""End of section from original"""

# DASH APP
app = Dash(__name__)

# Get all numeric columns from the dataframe
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Split columns into groups of 5
col_groups = [numeric_cols[i:i + 5] for i in range(0, len(numeric_cols), 5)]

# Define default selected features based on cols_to_check
default_selected_features = [col for col in cols_to_check if col in numeric_cols]

app.layout = html.Div(
  [
    html.H1("Board Game Data Correlation Heatmap"),
    html.Div(
      [
        html.Label("Select Features to Display:"),
        html.Div(
          [
            dcc.Checklist(
              id=f"feature-checklist-{i}",
              options=[{"label": col, "value": col} for col in group],
              value=[col for col in group if col in default_selected_features],  # Default selected features
              style={'display': 'inline-block', 'margin-right': '50px'}
            ) for i, group in enumerate(col_groups)
          ],
          style={'display': 'flex', 'flexWrap': 'wrap'}
        ),
        html.Br(),
        html.Button(
          "Toggle Between Heatmap/Pair Plot", id="toggle-button", n_clicks=0
        ),
      ],
      style={"margin": "10px"},
    ),
    dcc.Store(id="graph-state-store", data="heatmap"),  # Initial state is 'heatmap'
    dcc.Graph(id="graph-container"),
  ]
)

# Helper functions to build figures dynamically from selected features
def make_pairplot(df, features):
  n = len(features)
  # create an n x n grid of subplots
  fig = make_subplots(
    rows=n,
    cols=n,
    horizontal_spacing=0.02,
    vertical_spacing=0.02,
  )

  # add traces: histogram on diagonal, scatter on off-diagonals
  for i, y_col in enumerate(features):
    for j, x_col in enumerate(features):
      row = i + 1
      col = j + 1
      if i == j:
        # diagonal: histogram of the single feature
        fig.add_trace(
          go.Histogram(
            x=df[y_col].dropna(),
            nbinsx=30,
            marker=dict(color="lightgrey"),
            showlegend=False,
          ),
          row=row,
          col=col,
        )
        # Remove y-axis and x-axis titles for diagonal plots
        fig.update_yaxes(title_text="", row=row, col=col)
        fig.update_xaxes(title_text="", row=row, col=col)
      else:
        # off-diagonal: scatter of x vs y
        fig.add_trace(
          go.Scattergl(
            x=df[x_col].dropna(),
            y=df[y_col].dropna(),
            mode="markers",
            marker=dict(size=4, opacity=0.6),
            showlegend=False,
            hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
          ),
          row=row,
          col=col,
        )
        # label only outer axes to reduce clutter
        if row == n:
          fig.update_xaxes(title_text=x_col, row=row, col=col)
        if col == 1:
          fig.update_yaxes(title_text=y_col, row=row, col=col)

  fig.update_layout(
    title="Pair Plot of Board Game Features",
    height=min(300 * n, 1200),
    width=min(300 * n, 1200),
    margin=dict(l=40, r=40, t=60, b=40),
  )
  return fig


def make_heatmap(df, features):
    corr_matrix = df[features].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=features,
            y=features,
            colorscale="RdBu",
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Correlation Matrix of Board Game Features",
        xaxis={"side": "bottom"},
        yaxis={"side": "left"},
        height=800,
    )
    return fig


def empty_fig(message):
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
    )
    return fig


@app.callback(
    Output("graph-state-store", "data"),
    Input("toggle-button", "n_clicks"),
    State("graph-state-store", "data"),
    prevent_initial_call=True,  # Don't run this on page load
)
def update_graph_state(n_clicks, current_state):
    # If the button is clicked, toggle the state
    if current_state == "pairplot":
        return "heatmap"
    else:
        return "pairplot"


# build Inputs list dynamically to match the numbered checklist ids
_inputs = [Input("graph-state-store", "data")] + [
  Input(f"feature-checklist-{i}", "value") for i in range(len(col_groups))
]

@app.callback(
  Output("graph-container", "figure"),
  *_inputs,
)
def update_graph_figure(graph_state, *checklist_values):
  # flatten and dedupe selected features while preserving order
  selected_features = []
  for val in checklist_values:
    if val:
      selected_features.extend(val)
  selected_features = list(dict.fromkeys(selected_features))

  if not selected_features:
    return empty_fig("Please select at least one feature.")

  if graph_state == "pairplot":
    if len(selected_features) < 2:
      return empty_fig("Select at least two features for the pair plot.")
    return make_pairplot(df_clean, selected_features)
  else:  # heatmap
    if len(selected_features) < 2:
      return empty_fig("Select at least two features for the heatmap.")
    return make_heatmap(df_clean, selected_features)


if __name__ == "__main__":
    app.run(debug=True)
