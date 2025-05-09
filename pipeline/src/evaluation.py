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
    metrics: Input[Metrics]
):
    import logging
    import pandas as pd

    logging("Evaluation metrics:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")
        
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("gs://sentivista-453008_cloudbuild/metrics/evaluation_metrics.csv", index=False)
    logging.info("Metrics saved to GCS.")