from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1)
}

with DAG('feature_pipeline', 
         schedule_interval='@weekly',
         default_args=default_args) as dag:

    @task
    def preprocess_data():
        s3 = S3Hook(aws_conn_id='aws_default')
        
        # Get raw data
        raw_data = s3.read_key(
            bucket_name='xplode-raw',
            key='data.csv'
        )
        
        # Add preprocessing logic
        processed_data = raw_data.replace('http\S+', '', regex=True)
        
        # Upload processed data
        s3.load_string(
            string_data=processed_data.to_csv(),
            key='processed/data.csv',
            bucket_name='xplode-features'
        )

    @task
    def trigger_training():
        # Webhook to GitHub Actions
        import requests
        requests.post(
            'https://api.github.com/repos/your/repo/dispatches',
            json={'event_type': 'trigger-training'},
            headers={'Authorization': 'token YOUR_GITHUB_TOKEN'}
        )

    preprocess_data() >> trigger_training()