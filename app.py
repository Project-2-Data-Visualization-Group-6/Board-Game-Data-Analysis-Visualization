import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, dash, callback_context
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
app.layout = html.Div(
    [
        html.H1("Board Game Data Explorer and Predictor"),
        html.Div(
            [
                html.Label("Select Features for Analysis:"),
                dcc.Dropdown(
                    id="feature-checklist",
                    options=[{"label": col, "value": col} for col in cols_to_check],
                    value=cols_to_check,
                    multi=True,
                    placeholder="Select one or more features",
                    clearable=False,
                    style={"minWidth": "300px"},
                ),
                html.Br(),
                html.Label("Select Board Games to Include:"),
                dcc.Checklist(
                    id="game-filter",
                    options=[{"label": game, "value": game} for game in sorted(df_clean["boardgame"].unique())],
                    value=list(df_clean["boardgame"].unique()),
                    style={"maxHeight": "200px", "overflowY": "auto", "border": "1px solid #ccc", "padding": "10px"},
                ),
                html.Div([
                    html.Button("Select All Games", id="select-all-games", n_clicks=0, style={"marginRight": "10px"}),
                    html.Button("Deselect All Games", id="deselect-all-games", n_clicks=0),
                ], style={"marginTop": "5px"}),
                html.Br(),
                html.Button(
                    "Toggle Graph Type", id="toggle-button", n_clicks=0
                ),
                html.Div(
                    id="current-mode-label",
                ),
            ],
            style={"margin": "10px"},
        ),
        dcc.Store(id="graph-state-store", data="heatmap"),
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
                # Create custom hover text with game names
                hover_text = [
                    f"{game}<br>{x_col}: {x_val}<br>{y_col}: {y_val}"
                    for game, x_val, y_val in zip(df["boardgame"], df[x_col], df[y_col])
                ]
                
                fig.add_trace(
                    go.Scattergl(
                        x=df[x_col],
                        y=df[y_col],
                        mode="markers",
                        marker=dict(size=4, opacity=0.6),
                        showlegend=False,
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
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


def make_prediction_plot(df, features):
    target = "avg_rating"
    features = [f for f in features if f != target]
    if len(features) < 1:
        return empty_fig("Select at least one feature (excluding avg_rating) for prediction.")

    cols_needed = features + [target]
    if "boardgame" in df.columns:
        cols_needed = ["boardgame"] + cols_needed
    
    df_model = df[cols_needed].dropna()
    
    X = df_model[features]
    y = df_model[target]
    
    if "boardgame" in df_model.columns:
        game_names = df_model["boardgame"]
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
    Output("current-mode-label", "children"),
    Input("toggle-button", "n_clicks"),
    State("graph-state-store", "data"),
    prevent_initial_call=False,
)
def update_graph_state(n_clicks, current_state):
    modes = ["heatmap", "pairplot", "prediction"]
    next_state = modes[(modes.index(current_state) + 1) % len(modes)]
    mode_labels = {
        "heatmap": "Current Mode: Correlation Heatmap",
        "pairplot": "Current Mode: Pair Plot",
        "prediction": "Current Mode: Prediction Scatter (Linear Regression)",
    }
    return next_state, mode_labels[next_state]


@app.callback(
    Output("game-filter", "value"),
    Input("select-all-games", "n_clicks"),
    Input("deselect-all-games", "n_clicks"),
    State("game-filter", "options"),
    prevent_initial_call=True,
)
def update_game_selection(select_all_clicks, deselect_all_clicks, all_options):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "select-all-games":
        return [option["value"] for option in all_options]
    elif button_id == "deselect-all-games":
        return []
    
    return dash.no_update


@app.callback(
    Output("graph-container", "figure"),
    Input("graph-state-store", "data"),
    Input("feature-checklist", "value"),
    Input("game-filter", "value"),
)
def update_graph_figure(graph_state, selected_features, selected_games):
    if not selected_features:
        return empty_fig("Please select at least one feature.")
    
    if not selected_games:
        return empty_fig("Please select at least 3 board games.")
    
    if len(selected_games) < 3:
        return empty_fig(f"Please select at least 3 board games. Currently selected: {len(selected_games)}")
    
    df_filtered = df_clean[df_clean["boardgame"].isin(selected_games)]
    
    if len(df_filtered) == 0:
        return empty_fig("No data available for selected games.")
    
    if graph_state == "pairplot":
        if len(selected_features) < 2:
            return empty_fig("Select at least two features for the pair plot.")
        return make_pairplot(df_filtered, selected_features)

    elif graph_state == "prediction":
        return make_prediction_plot(df_filtered, selected_features)

    else: # heatmap
        if len(selected_features) < 2:
            return empty_fig("Select at least two features for the heatmap.")
        return make_heatmap(df_filtered, selected_features)


if __name__ == "__main__":
    app.run(debug=True)