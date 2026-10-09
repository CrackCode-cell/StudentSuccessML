# StudentSuccessML

A beginner-friendly machine learning project that explores patterns in
student-related data using Python.

StudentSuccessML combines data analysis, machine learning, visualizations,
and interactive dashboards in one project.

The goal is to learn how data can be collected, explored, used to train
models, and turned into useful information.

## What Does This Project Do?

The project uses a small dataset containing these features:

- Study hours
- Attendance
- Previous GPA
- Assignments completed
- Sleep hours
- Success label (0 or 1)

The project explores relationships between these features and the success
label. It trains machine learning models, compares their results, and
allows users to experiment with different inputs.

The included dataset is synthetic, meaning it is generated for learning.
It does not represent real students.

## Main Features

### 1. Data Analysis

Explore the dataset using pandas, summary statistics, charts, and
correlation analysis.

### 2. Logistic Regression

Train a supervised machine learning model to predict the success label
from the input features.

### 3. Random Forest

Train a second model and compare its results with Logistic Regression.

### 4. Model Evaluation

Review metrics such as accuracy, precision, recall, and F1 score.

### 5. Feature Importance

Explore which features the trained Random Forest uses most when making
predictions.

Feature importance does not prove that a feature causes an outcome.

### 6. K-Means Clustering

Use unsupervised machine learning to group records with similar
characteristics without using the success label to create the groups.

### 7. Interactive Dashboards

Use Streamlit dashboards to explore the dataset, compare model results,
experiment with predictions, and visualize clusters.

### 8. Prediction and Study Advice

Try individual predictions and explore basic rule-based suggestions.

The study advisor is rule-based software, not a generative AI chatbot.

## Technology Used

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- joblib
- Streamlit
- Plotly
- Git and GitHub

## Project Structure

```text
StudentSuccessML/
├── data/
│   └── student_data.csv
├── src/
│   ├── analysis.py
│   ├── train_model.py
│   ├── predict.py
│   ├── ai_advisor.py
│   ├── dashboard.py
│   ├── cluster_dashboard.py
│   ├── generate_dataset.py
│   ├── model_experiments.py
│   ├── learning_curve.py
│   ├── error_analysis.py
│   └── compare_models.py
├── docs/
│   ├── PROJECT_GUIDE.md
│   └── ML_CONCEPTS.md
├── requirements.txt
├── README.md
└── .gitignore
```

Some scripts generate additional files when they run. Those generated files
may not be included in Git.

## Getting Started

### Requirements

Install Python 3 and Git. A recent Python 3 version is recommended.

### 1. Download the project

If you have not cloned the repository yet:

```bash
git clone https://github.com/CrackCode-cell/StudentSuccessML.git
cd StudentSuccessML
```

If you already have the project on your computer, open Terminal and run:

```bash
cd ~/StudentSuccessML
```

### 2. Create a virtual environment

On macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

A virtual environment keeps this project's Python packages separate
from packages used by other projects.

### 3. Install the required packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Explore the data

```bash
python3 src/analysis.py
```

This script prints summary information and generates visualizations.

### 5. Train the main model

```bash
python3 src/train_model.py
```

This trains Logistic Regression, evaluates it, and saves the model file
used by the prediction tools.

### 6. Compare machine learning models

```bash
python3 src/model_experiments.py
```

This compares Logistic Regression with Random Forest and generates
feature-importance results.

### 7. Open the main dashboard

```bash
python3 -m streamlit run src/dashboard.py
```

Streamlit will show a local web address in Terminal. Open that address
in your browser.

### 8. Open the clustering dashboard

In another Terminal tab or after stopping the first dashboard, run:

```bash
python3 -m streamlit run src/cluster_dashboard.py
```

This dashboard lets you explore K-Means clustering.

## Other Experiments

Generate a new synthetic dataset:

```bash
python3 src/generate_dataset.py --rows 500 --noise 0.10
```

Important: this replaces `data/student_data.csv`. Back up your existing
dataset first if you want to preserve it.

Run the learning-curve experiment:

```bash
python3 src/learning_curve.py
```

Inspect incorrect predictions:

```bash
python3 src/error_analysis.py
```

Try a single prediction:

```bash
python3 src/predict.py
```

Try the rule-based advisor:

```bash
python3 src/ai_advisor.py
```

If you use `compare_models.py`, follow its instructions and verify that
the script exists in your local `src/` directory.

## How the Project Works

1. Load the data with pandas.
2. Explore patterns using tables and charts.
3. Select input features and the target label.
4. Train a machine learning model.
5. Evaluate predictions on held-out data.
6. Compare models and inspect feature importance.
7. Use dashboards to explore the results.

Read `docs/PROJECT_GUIDE.md` for a more detailed explanation.

## Important Limitations

- The dataset is synthetic and intended for learning.
- Model scores on synthetic data do not show how accurately the models
  would predict real students' outcomes.
- Small datasets can produce unstable evaluation scores.
- Correlation and feature importance do not prove causation.
- K-Means discovers mathematical groupings; it does not discover
  objectively correct types of students.
- Predictions should not be used to label, rank, or make important
  decisions about real students.

## Learning Goals

This project is designed to practice:

- Data cleaning and exploration
- Basic statistics and visualization
- Supervised machine learning
- Unsupervised machine learning
- Model evaluation
- Feature importance
- Python scripting
- Interactive dashboard development
- Git and GitHub workflows

## Future Improvements

Possible future additions include:

- More models and stronger evaluation methods
- Better dashboard explanations
- Model interpretation tools
- A conversational study assistant powered by a language model
- More realistic, ethically sourced data for carefully designed research

## Author

Created as a hands-on learning project to explore Python, data analysis,
and machine learning.

