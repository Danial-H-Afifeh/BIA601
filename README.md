# GA Feature Selection Web App

A web application for genetic algorithm-based feature selection, built with Flask and scikit-learn.

## Features

- Upload CSV files for analysis
- Select target column for prediction
- Choose between Logistic Regression and Random Forest models
- Configure GA parameters (population size, generations, alpha, beta)
- View selected features and performance metrics
- RTL interface support

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd ga-feature-selection
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python web_app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload a CSV file and configure your settings:
   - Select your target column
   - Choose a model (Logistic Regression or Random Forest)
   - Adjust GA parameters if needed
   - Click "تشغيل" to run the analysis

4. View results including:
   - Cross-validation accuracy
   - Selected features ratio
   - List of selected features

Results are automatically saved to `results/web_last_result.json`

## Project Structure

- `web_app.py`: Main Flask application
- `ga_feature_selection.py`: GA implementation
- `static/css/style.css`: RTL-aware styling
- `uploads/`: Temporary storage for uploaded files
- `results/`: JSON output storage

## License

MIT License