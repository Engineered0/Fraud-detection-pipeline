import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Define the dataset and model paths
DATASET_PATH = "/usr/local/airflow/dags/creditcard.csv"
MODEL_PATH = "/usr/local/airflow/dags/fraud_detection_model.pkl"

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "retries": 2,  # Set retries to 2 or more
    "retry_delay": timedelta(minutes=5),  # Delay between retries
}

# Step 1: Extract
def extract_data(**kwargs):
    print("Extracting data...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Data loaded successfully with shape: {df.shape}")
    kwargs['ti'].xcom_push(key='extracted_data', value=df.to_json())

# Step 2: Transform
def transform_data(**kwargs):
    print("Transforming data...")
    ti = kwargs['ti']
    df_json = ti.xcom_pull(task_ids='extract_data', key='extracted_data')
    df = pd.read_json(df_json)

    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Is_Peak_Hour'] = df['Hour'].apply(lambda x: 1 if 18 <= x <= 23 or 0 <= x <= 6 else 0)
    df = df.drop(['Time'], axis=1)

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    kwargs['ti'].xcom_push(key='train_test_data', value={
        'X_train': X_train.to_json(),
        'X_test': X_test.to_json(),
        'y_train': y_train.to_json(),
        'y_test': y_test.to_json()
    })

# Step 3: Load
def load_model(**kwargs):
    print("Training the model...")
    ti = kwargs['ti']
    train_test_data = ti.xcom_pull(task_ids='transform_data', key='train_test_data')

    X_train = pd.read_json(train_test_data['X_train'])
    X_test = pd.read_json(train_test_data['X_test'])
    y_train = pd.read_json(train_test_data['y_train'])
    y_test = pd.read_json(train_test_data['y_test'])

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    print("Model training completed.")

    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Define the DAG
with DAG(
    dag_id="fraud_detection_pipeline",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["fraud-detection", "machine-learning"],  # Add tags
) as dag:

    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
        provide_context=True,
    )

    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
        provide_context=True,
    )

    load_task = PythonOperator(
        task_id="load_model",
        python_callable=load_model,
        provide_context=True,
    )

    extract_task >> transform_task >> load_task
