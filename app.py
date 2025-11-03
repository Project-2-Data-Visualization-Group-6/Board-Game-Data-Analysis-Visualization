import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    html.H1("Board Game Data Explorer and Predictor"),
    html.Div(
      [
        html.Label("Select Features for Analysis:"),
        html.Div(
          [
            dcc.Checklist(
              id=f"feature-checklist-{i}",
              options=[{"label": col, "value": col} for col in group],
              value=[col for col in group if col in default_selected_features],
              style={'display': 'inline-block', 'margin-right': '50px'}
            ) for i, group in enumerate(col_groups)
          ],
          style={'display': 'flex', 'flexWrap': 'wrap'}
        ),
        html.Br(),
        html.Div(
          [
            html.Button("Heatmap", id="heatmap-button", n_clicks=0),
            html.Button("Pair Plot", id="pairplot-button", n_clicks=0),
            html.Button("Prediction Plot", id="prediction-button", n_clicks=0),
          ],
          style={"margin": "10px"}
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
  if n < 2:
    return empty_fig("Select at least two features for the pair plot.")

  CELL_PX = 260            # pixel size for each small subplot (keep cells consistent)
  if n < 6:
    GAP_FRAC = 0.03         # spacing between subplots (fraction of cell size)
  elif n < 10:
    GAP_FRAC = 0.02         # tighter spacing for larger n
  else:
    GAP_FRAC = 0.01
  MARGINS = dict(l=70, r=40, t=70, b=60)  # padding to avoid clipping

  # overall figure grows with n, while each cell remains CELL_PX
  fig_width  = n * CELL_PX + MARGINS["l"] + MARGINS["r"]
  fig_height = n * CELL_PX + MARGINS["t"] + MARGINS["b"]

  # build n x n grid
  fig = make_subplots(
    rows=n,
    cols=n,
    horizontal_spacing=GAP_FRAC,
    vertical_spacing=GAP_FRAC,
  )

  # fill the grid
  for i, y_col in enumerate(features):
    for j, x_col in enumerate(features):
      row, col = i + 1, j + 1

      if i == j:
        # diagonal: histogram
        fig.add_trace(
          go.Histogram(
            x=df[y_col].dropna(),
            nbinsx=30,
            marker=dict(opacity=0.85),
            showlegend=False,
          ),
          row=row, col=col
        )
      else:
        # off-diagonal: scatter
        fig.add_trace(
          go.Scattergl(
            x=df[x_col].dropna(),
            y=df[y_col].dropna(),
            mode="markers",
            marker=dict(size=4, opacity=0.6),
            hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
            showlegend=False,
          ),
          row=row, col=col
        )

      if row == n:
        fig.update_xaxes(title_text=x_col, row=row, col=col)
      if col == 1:
        fig.update_yaxes(title_text=y_col, row=row, col=col)

  # layout: grow with n; keep cells constant size
  fig.update_layout(
    title="Pair Plot of Board Game Features",
    autosize=False,                 # honor explicit pixels
    width=fig_width,
    height=fig_height,
    margin=MARGINS,
  )

  # prevent clipping and make titles readable on outer edges
  fig.update_xaxes(automargin=True, title_standoff=6)
  fig.update_yaxes(automargin=True, title_standoff=8)

  return fig



# heatmap
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
  
# Makes the prediction plot (Sal)
def make_prediction_plot(df, features):
    target = "avg_rating"
    features = [f for f in features if f != target]
    if len(features) < 1:
        return empty_fig("Select at least one feature (excluding avg_rating) for prediction.")

    # Include boardgame column in the dataframe for splitting
    cols_needed = features + [target]
    if 'boardgame' in df.columns:
        cols_needed = ['boardgame'] + cols_needed
    
    df_model = df[cols_needed].dropna()
    
    X = df_model[features]
    y = df_model[target]
    
    # Split with game names if available
    if 'boardgame' in df_model.columns:
        game_names = df_model['boardgame']
        X_train, X_test, y_train, y_test, _, game_names_test = train_test_split(
            X, y, game_names, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        game_names_test = None
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

    fig = go.Figure()
    
    # Create hover text with game names
    if game_names_test is not None:
        hover_text = [
            f"{game}<br>Actual: {actual:.2f}<br>Predicted: {pred:.2f}"
            for game, actual, pred in zip(game_names_test, y_test, predictions)
        ]
        hovertemplate = "%{text}<extra></extra>"
    else:
        hover_text = None
        hovertemplate = "Actual: %{x}<br>Predicted: %{y}<extra></extra>"
    
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=predictions,
            mode="markers",
            marker=dict(color="skyblue", size=6, opacity=0.6),
            name="Predicted vs Actual",
            text=hover_text,
            hovertemplate=hovertemplate,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Ideal Fit",
        )
    )

    fig.update_layout(
        title=f"Linear Regression Prediction (R² = {r2:.3f})",
        xaxis_title="Actual Average Rating",
        yaxis_title="Predicted Average Rating",
        height=700,
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
  Input("heatmap-button", "n_clicks_timestamp"),
  Input("pairplot-button", "n_clicks_timestamp"),
  Input("prediction-button", "n_clicks_timestamp"),
  # Input("boxplot-button", "n_clicks_timestamp"),
  # Input("countplot-button", "n_clicks_timestamp"),
  # Input("histogram-button", "n_clicks_timestamp"),
  State("graph-state-store", "data"),
  prevent_initial_call=True,
)
def update_graph_state(
  heatmap_ts,
  pairplot_ts,
  prediction_ts,
  # boxplot_ts,
  # countplot_ts,
  # histogram_ts,
  current_state,
):
  # Map button keys to their last-click timestamps
  ts_map = {
    "heatmap": heatmap_ts,
    "pairplot": pairplot_ts,
    "prediction": prediction_ts,
    # "boxplot": boxplot_ts,
    # "countplot": countplot_ts,
    # "histogram": histogram_ts,
  }

  # Keep only buttons that have been clicked (timestamp not None)
  clicked = {k: v for k, v in ts_map.items() if v is not None}
  if not clicked:
    # no clicks to handle, keep current state
    return current_state

  # Return the state corresponding to the most recently clicked button
  most_recent = max(clicked.items(), key=lambda kv: kv[1])[0]
  return most_recent


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
  if graph_state == "prediction":
    # if len(selected_features) < 2:
    #   return empty_fig("Select at least two features for the pair plot.")
    return make_prediction_plot(df_clean, selected_features)
  else:  # heatmap
    if len(selected_features) < 2:
      return empty_fig("Select at least two features for the heatmap.")
    return make_heatmap(df_clean, selected_features)


if __name__ == "__main__":
    app.run(debug=True)