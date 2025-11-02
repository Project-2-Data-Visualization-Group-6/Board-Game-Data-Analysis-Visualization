import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output

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
app.layout = html.Div([
  html.H1('Board Game Data Correlation Heatmap'),
  html.Div([
    html.Label('Select Features to Display:'),
    dcc.Checklist(
      id='feature-checklist',
      options=[{'label': col, 'value': col} for col in cols_to_check],
      value=cols_to_check,  # All features selected by default
      inline=True
    )
  ], style={'margin': '10px'}),
  dcc.Graph(id='correlation-heatmap')
])

@app.callback(
  Output('correlation-heatmap', 'figure'),
  Input('feature-checklist', 'value')
)
def update_heatmap(selected_features):
  if not selected_features:  # If no features selected
    return {}
  
  corr_matrix = df_clean[selected_features].corr()
  
  return {
    'data': [
      go.Heatmap(
        z=corr_matrix,
        x=selected_features,
        y=selected_features,
        colorscale='RdBu',
        text=corr_matrix.round(2),  # Show correlation values
        texttemplate='%{text}',     # Display the text
        textfont={"size": 10},      # Adjust font size
        hoverongaps=False
      )
    ],
    'layout': {
      'title': 'Correlation Matrix of Board Game Features',
      'xaxis': {'side': 'bottom'},
      'yaxis': {'side': 'left'},
      'height': 800
    }
  }

if __name__ == "__main__":
  app.run(debug=True)
