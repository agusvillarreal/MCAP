from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution(data, column, title="Distribution"):
    """Plots the distribution of a column."""
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{column}_distribution.png')
    print(f"Distribution plot saved as {column}_distribution.png")

def plot_correlation_heatmap(df, title="Correlation Heatmap"):
    """Plots a correlation heatmap for numerical columns."""
    plt.figure(figsize=(10, 8))
    # Select only numerical columns for correlation
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numerical_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("Correlation heatmap saved as correlation_heatmap.png")

def train_model(X_train, y_train, preprocessor, model_type='linear_regression'):
    """Trains a model using a pipeline with the given preprocessor."""
    if model_type == 'linear_regression':
        regressor = LinearRegression()
    elif model_type == 'random_forest':
        regressor = RandomForestRegressor(random_state=42)
    elif model_type == 'gradient_boosting':
        regressor = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose 'linear_regression', 'random_forest', or 'gradient_boosting'.")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', regressor)])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def calculate_rse(y_true, y_pred, p):
    """Calculates the Residual Standard Error."""
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    rse = np.sqrt(rss / (n - p - 1))
    return rse

def evaluate_model(model, X_test, y_test, model_name="Model", p=None):
    """Evaluates the model and prints metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- {model_name} Performance ---")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    if p is not None:
        rse = calculate_rse(y_test, y_pred, p)
        print(f"Residual Standard Error (RSE): {rse:.2f}")
        return mse, r2, rse, y_pred
    
    return mse, r2, y_pred

def plot_residuals(y_test, y_pred, model_name="Model"):
    """Plots residuals vs predicted values."""
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Scores')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Residual Plot')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_residuals.png')
    print(f"Residual plot saved as {model_name.lower().replace(' ', '_')}_residuals.png")

def plot_predictions(y_test, y_pred, model_name="Model"):
    """Plots actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_predictions.png')
    print(f"Plot saved as {model_name.lower().replace(' ', '_')}_predictions.png")
