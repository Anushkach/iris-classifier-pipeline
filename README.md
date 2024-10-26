```markdown
# Iris MLflow Project

This project demonstrates how to set up and run an ML pipeline with MLflow tracking using the Iris dataset.

## Setup

### 1. Create a new project directory and virtual environment

```bash
# Create project directory
mkdir iris_mlflow_project
cd iris_mlflow_project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install mlflow scikit-learn pandas numpy schedule
```

### 2. Create necessary directories

```bash
# Create directories for models and mlflow
mkdir models
mkdir mlruns
```

### 3. Save the Python script

Save the provided Python script as `iris_pipeline.py`.

### 4. Start the MLflow tracking server

```bash
# Start MLflow server
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

This will start the MLflow UI server at [http://localhost:5000](http://localhost:5000).

### 5. Run the pipeline

In a new terminal (keep the MLflow server running), activate the virtual environment and run the pipeline:

```bash
# Activate virtual environment again (since we're in a new terminal)
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Run the pipeline
python 

iris_pipeline.py


```

### 6. Optional MLflow commands

```bash
# List all experiments
mlflow experiments list

# Get specific experiment details
mlflow experiments describe <experiment_id>

# List all runs for an experiment
mlflow runs list --experiment-id <experiment_id>

# Get specific run details
mlflow runs describe <run_id>
```

### 7. To stop everything

```bash
# Stop the Python script (if running continuously)
Ctrl + C

# Stop the MLflow server
Ctrl + C

# Deactivate virtual environment
deactivate
```

## Directory Structure

```
iris_mlflow_project/
├── venv/
├── models/
│   ├── iris_model.joblib
│   └── scaler.joblib
├── mlruns/
├── 

mlflow.db


└── 

iris_pipeline.py


```

## Common Issues and Solutions

### 1. If port 5000 is already in use

```bash
# Use a different port
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

### 2. To check if MLflow server is running

```bash
# Check processes using port 5000
lsof -i :5000
```

### 3. To kill MLflow server if it's stuck

```bash
# Find the process ID
ps aux | grep mlflow

# Kill the process
kill <process_id>
```

### 4. To clear all MLflow data and start fresh

```bash
# Remove MLflow database and runs
rm 

mlflow.db


rm -rf mlruns
```

### 5. To view logs in real-time

```bash
# In a new terminal
tail -f mlflow.log
```

## Best Practices

### 1. Always use virtual environment to avoid package conflicts

```bash
# Create new virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Deactivate when done
deactivate
```

### 2. Save package dependencies

```bash
# After installing all packages
pip freeze > requirements.txt

# To reinstall packages later
pip install -r requirements.txt
```

### 3. Run MLflow server in background (for production)

```bash
# Start server in background
nohup mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 > mlflow.log 2>&1 &

# To stop it later, find and kill the process
ps aux | grep mlflow
kill <process_id>
```
```

