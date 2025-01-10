# Fraud Detection Pipeline

This repository contains a machine learning pipeline for detecting fraudulent transactions using a Random Forest classifier. The pipeline is built using Apache Airflow for orchestration and includes data extraction, transformation, and model training.

## Overview

The fraud detection pipeline performs the following steps:

1. **Extract**: Load the dataset from a CSV file.
2. **Transform**: Preprocess the data, including feature scaling and engineering.
3. **Load**: Train a Random Forest model on the processed data and save the model for future use.

## Requirements

- **Python 3.x**
- **Pandas**
- **Scikit-learn**
- **Joblib**
- **Apache Airflow**

## Installation

1. Clone this repository:
git clone https://github.com/Engineered0/fraud-detection-pipeline.git
cd fraud-detection-pipeline

pip install -r requirements.txt


## Dataset

The dataset used in this pipeline is assumed to be located at `/usr/local/airflow/dags/creditcard.csv`. Make sure to place your dataset in this path or update the `DATASET_PATH` variable in the code accordingly.

## Usage

1. Start your Apache Airflow server.
2. Place the DAG file in your Airflow `dags` directory.
3. Trigger the `fraud_detection_pipeline` DAG from the Airflow UI.

## Output

The trained model will be saved as `fraud_detection_model.pkl` in the `/usr/local/airflow/dags/` directory upon successful completion of the pipeline.

## Example Output

The pipeline will output a classification report and confusion matrix after training, providing insights into model performance.

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements or enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
