from kfp.v2 import dsl
from kfp.v2.dsl import (
    Artifact,    # For handling ML artifacts
    Dataset,     # For handling datasets
    Input,       # For component inputs
    Model,       # For handling ML models
    Output,      # For component outputs
    Metrics,     # For tracking metrics
    HTML,        # For visualization
    component,   # For creating pipeline components
    pipeline     # For defining the pipeline
)
from kfp.v2 import compiler
from google.cloud.aiplatform import pipeline_jobs
from src.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE,
    output_component_file="evaluation.yaml"
)
def evaluation_op(
    model: Input[Model],
    preprocessed_dataset: Input[Dataset],
    metrics: Output[Metrics]
):
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    import logging

    model = joblib.load(model.path)
    df = pd.read_csv(preprocessed_dataset.path)
    
    X_test = df.drop(columns=['target'])
    y_test = df['target']

    predictions = model.predict(X_test)
    acc = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    df_metrics = pd.DataFrame({
        'Metric': ['MSE', 'R2'],
        'Value': [acc, r2]
    })
    df_metrics.to_csv(metrics.path, index=False)
    logging.info(f"Metrics saved to: {metrics.path}")