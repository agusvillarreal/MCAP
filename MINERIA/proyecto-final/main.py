import data_processing
import model
import pandas as pd

def main():
    print("Starting Student Performance Prediction Pipeline...")

    # 1. Load Data
    filepath = 'test_scores.xls'
    print(f"Loading data from {filepath}...")
    df = data_processing.load_data(filepath)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return

    # 1.1 Exploratory Data Analysis (New)
    print("Generating exploratory plots...")
    model.plot_distribution(df, 'posttest', title="Distribution of Posttest Scores")
    model.plot_correlation_heatmap(df, title="Correlation Heatmap of Numerical Features")

    # 2. Preprocess Data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = data_processing.preprocess_data(df)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Calculate number of predictors (p) for RSE
    # One-hot encoding increases the number of features, so we get it from the transformed training set
    X_train_transformed = preprocessor.fit_transform(X_train)
    p = X_train_transformed.shape[1]
    print(f"Number of predictors (p): {p}")

    # 3. Train and Evaluate Linear Regression
    print("\nTraining Linear Regression...")
    lr_pipeline = model.train_model(X_train, y_train, preprocessor, model_type='linear_regression')
    mse_lr, r2_lr, rse_lr, y_pred_lr = model.evaluate_model(lr_pipeline, X_test, y_test, model_name="Linear Regression", p=p)
    model.plot_predictions(y_test, y_pred_lr, model_name="Linear Regression")
    model.plot_residuals(y_test, y_pred_lr, model_name="Linear Regression")

    # 4. Train and Evaluate Random Forest
    print("\nTraining Random Forest...")
    rf_pipeline = model.train_model(X_train, y_train, preprocessor, model_type='random_forest')
    mse_rf, r2_rf, rse_rf, y_pred_rf = model.evaluate_model(rf_pipeline, X_test, y_test, model_name="Random Forest", p=p)
    model.plot_predictions(y_test, y_pred_rf, model_name="Random Forest")
    model.plot_residuals(y_test, y_pred_rf, model_name="Random Forest")

    # 5. Train and Evaluate Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb_pipeline = model.train_model(X_train, y_train, preprocessor, model_type='gradient_boosting')
    mse_gb, r2_gb, rse_gb, y_pred_gb = model.evaluate_model(gb_pipeline, X_test, y_test, model_name="Gradient Boosting", p=p)
    model.plot_predictions(y_test, y_pred_gb, model_name="Gradient Boosting")
    model.plot_residuals(y_test, y_pred_gb, model_name="Gradient Boosting")

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
