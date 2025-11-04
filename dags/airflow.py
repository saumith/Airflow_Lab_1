from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

default_args = {
    'owner': 'ML_Engineer',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'SVM_Classification_Pipeline',
    default_args=default_args,
    description='SVM Classification Pipeline',
    schedule=None,
    catchup=False,
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, 'svm_model.pkl'],
    dag=dag,
)

load_model_task = PythonOperator(
    task_id='load_model_predict_task',
    python_callable=load_model_elbow,
    op_args=['svm_model.pkl', build_save_model_task.output],
    dag=dag,
)

load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task