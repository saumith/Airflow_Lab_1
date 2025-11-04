# Airflow Lab 1 - SVM Classification Pipeline

Apache Airflow project for SVM classification pipeline with automated data loading, preprocessing, model training, and predictions.

## Overview

A 4-task DAG pipeline:
1. **load_data_task** - Load data from CSV
2. **data_preprocessing_task** - Clean & normalize data
3. **build_save_model_task** - Train SVM model
4. **load_model_predict_task** - Generate predictions

## Quick Start

### Prerequisites
- Docker Desktop (4GB+ memory)
- Docker Compose
- SSH key for Git

### Setup

```bash
# Clone repo
git clone git@github.com:saumith/Airflow_Lab_1.git
cd Airflow_Lab_1

# Initialize Airflow
mkdir -p ./config ./logs ./plugins ./working_data
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Start Airflow
docker compose up airflow-init
docker compose up
```

Visit `http://localhost:8080` → Login: `airflow/airflow` (default)

### Trigger DAG
1. Go to DAGs section
2. Find **SVM_Classification_Pipeline**
3. Click **Trigger DAG**

## Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python src/lab.py
```

## Dataset

**Adult Income Dataset** - Binary classification dataset predicting income levels

- **Source:** UCI Machine Learning Repository
- **Training data:** `data/file.csv` (36,177 samples)
- **Test data:** `data/test.csv` 
- **Features:** 9 attributes (age, education, occupation, etc.)
- **Target:** Binary classification (0: ≤$50K, 1: >$50K)

### Download Data

The dataset is automatically included in the repository. If you need to regenerate it:

```bash
# Run the data preparation script
python adult_income_data.py
```

This creates/updates the CSV files in the `data/` directory.

## Project Structure

```
Airflow_Lab_1/
├── dags/airflow.py           # DAG definition
├── src/lab.py                # ML functions
├── data/
│   ├── file.csv              # Training data
│   └── test.csv              # Test data
├── docker-compose.yaml       # Docker config
├── requirements.txt          # Dependencies
└── README.md
```

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.8343 |
| Precision | 0.8201 |
| Recall | 0.8343 |
| F1-Score | 0.8120 |

**Dataset:** 36,177 training samples, 9 features, binary classification

## Airflow Execution

### DAG Graph
![DAG Graph](.assets/airflow_dag_graph.png)

All tasks execute successfully in sequence.

### Task Instances
![Task Instances](.assets/airflow_task_instances.png.png)

Detailed execution metrics for each task.

## Output Files

- `model/svm_model.pkl` - Trained SVM
- `model/scaler.pkl` - MinMaxScaler
- `logs/` - Execution logs

## Troubleshooting

**DAG not appearing?** Check logs:
```bash
docker compose logs webserver
```

**Out of memory?** Allocate more memory in Docker Desktop

**Permission error?** Ensure `.env` has correct UID:
```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

## References

- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)
- [ML with Ramin - Airflow Lab 1](https://www.mlwithramin.com/blog/airflow-lab1)

## License

Educational purposes
