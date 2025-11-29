# Student Performance Prediction

This project aims to predict student test scores based on various features such as school setting, teaching method, and pretest scores.

## Project Structure
- `main.py`: Main script to run the prediction pipeline.
- `data_processing.py`: Functions for data loading and preprocessing.
- `model.py`: Functions for model training and evaluation.
- `test_scores.xls`: The dataset (CSV format).
- `requirements.txt`: Python dependencies.

## Setup
1. Initialize a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script:
```bash
python main.py
```

## Methodology
1. **Data Preparation**:
   - Loaded data from `test_scores.xls`.
   - Dropped `student_id`.
   - One-hot encoded categorical variables.
   - Scaled numerical variables.
   - Split data into training (80%) and testing (20%) sets.

2. **Model Construction**:
   - **Linear Regression**: A baseline linear model.
   - **Random Forest**: An ensemble method to capture non-linear relationships.

3. **Validation**:
   - Evaluated models using Mean Squared Error (MSE) and R-squared ($R^2$).
   - Visualized actual vs. predicted scores.
