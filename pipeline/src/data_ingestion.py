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
    output_component_file="data_ingestion.yaml"
)
def data_ingestion_op(
        dataset: Output[Dataset]
    ):
    import pandas as pd
    import logging
    import gcsfs
    
    logging.basicConfig(level=logging.INFO)
    
    logging.info("Starting data ingestion...")
    
    #A CHANGER EN FONCTION DE COMMENT ON STORE NOS DATA
    GCS_BUCKET = "gs://sentivista-453008_cloudbuild"
    FILE_PATH = f"{GCS_BUCKET}/data/sentivista.csv"
    
    logging.info(f"Loading dataset from {FILE_PATH}...")
    fs = gcsfs.GCSFileSystem()
    with fs.open(FILE_PATH, 'r') as f:
        df = pd.read_csv(f)
        
    
    logging.info(f"Saving dataset to {dataset.path}...")
    df.to_csv(dataset.path, index=False)
    
    logging.info("Data ingestion completed successfully.")
