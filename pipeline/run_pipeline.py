from src.config import *

from kfp.v2.dsl import (
    pipeline     # For defining the pipeline
)
from kfp.v2 import compiler
from google.cloud import aiplatform

from src.data_ingestion import data_ingestion
from src.preprocessing import preprocessing
from src.training import training
from src.evaluation import evaluation

@pipeline(
    name="houseprice_pipeline",
    pipeline_root=PIPELINE_ROOT
)
def sentivista():
    # Define the components
    ingestion_task = data_ingestion()
    
    preprocessing_task = preprocessing(
        input_dataset=ingestion_task.outputs["dataset"]
    )
    
    training_task = training(
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"],
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    )
    
    evaluation_task = evaluation(
        model=training_task.outputs["model"],
        test_set=training_task.outputs["test_set"]
    )

compiler.Compiler().compile(
    pipeline_func=sentivista,
    package_path='sentivista.json'
)

aiplatform.init(project=PROJECT_ID, location=REGION)

pipeline_job = aiplatform.PipelineJob(
    display_name="sentivista",
    template_path="sentivista.json",
    pipeline_root=PIPELINE_ROOT
)

pipeline_job.run()