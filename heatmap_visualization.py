import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
csv_file = 'boardgame-geek-dataset_organized.csv'
df = pd.read_csv(csv_file, encoding='utf-8')

print(f'DataFrame shape: {df.shape}')
print('Data loaded successfully.')

# Data cleaning - drop rows with missing critical values
cols_to_check = ['min_players', 'max_players', 'min_playtime', 'max_playtime', 
                 'minimum_age', 'avg_rating', 'num_ratings', 'complexity']
df_clean = df.dropna(subset=cols_to_check)

print(f'Shape after cleaning: {df_clean.shape}')

# Create a subset with only numeric data for correlation analysis
numeric_df = df_clean.select_dtypes(include=[np.number])

if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(12,10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print('Heatmap saved as correlation_heatmap.png')
    plt.show()

# Pair Plot for a sample of numeric columns (this can be verbose with many columns)
sample_cols = ['avg_rating', 'num_ratings', 'complexity', 'min_playtime', 'max_playtime']
sns.pairplot(df_clean[sample_cols])
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.savefig('pair_plot.png', dpi=300, bbox_inches='tight')
print('Pair plot saved as pair_plot.png')
plt.show()

# Histogram for the complexity distribution
plt.figure(figsize=(8,4))
sns.histplot(df_clean['complexity'].dropna(), kde=True)
plt.title("Distribution of Game Complexity")
plt.tight_layout()
plt.savefig('complexity_distribution.png', dpi=300, bbox_inches='tight')
print('Complexity histogram saved as complexity_distribution.png')
plt.show()

# Count plot for minimum players
plt.figure(figsize=(10,4))
sns.countplot(x='min_players', data=df_clean)
plt.title("Distribution of Minimum Players")
plt.tight_layout()
plt.savefig('min_players_distribution.png', dpi=300, bbox_inches='tight')
print('Min players count plot saved as min_players_distribution.png')
plt.show()

# Box plot for average ratings
plt.figure(figsize=(8,4))
sns.boxplot(x=df_clean['avg_rating'])
plt.title("Boxplot of Average Rating")
plt.tight_layout()
plt.savefig('avg_rating_boxplot.png', dpi=300, bbox_inches='tight')
print('Average rating boxplot saved as avg_rating_boxplot.png')
plt.show()

print('\nAll visualizations completed!')

