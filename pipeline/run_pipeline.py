from src.config import *

from kfp.v2.dsl        import pipeline
from kfp.v2            import compiler
from google.cloud      import aiplatform

from src.data_ingestion import data_ingestion_op
from src.preprocessing import preprocessing_op
from src.training      import training_op
from src.evaluation    import evaluation_op

@pipeline(
    name="sentivista-pipeline",
    pipeline_root=PIPELINE_ROOT,
    parameters={
        "raw_data_path":        "gs://sentivista-453008_cloudbuild/data/sentivista.csv",
        "roberta_checkpoint":   "gs://sentivista-453008_cloudbuild/models/roberta/roberta_full_model/",
        "output_model_path":    "gs://sentivista-453008_cloudbuild/models/metrics/"
    }
)
def sentivista(
    raw_data_path: str,
    roberta_checkpoint: str,
    output_model_path: str
):
    ingestion     = data_ingestion_op(
        gcs_path=raw_data_path
    )
    
    preprocessing = preprocessing_op(
        input_dataset=ingestion.outputs["dataset"]
    )
    training      = training_op(
        model_checkpoint=roberta_checkpoint,
        preprocessed_dataset=preprocessing.outputs["preprocessed_dataset"]
    )
    evaluation    = evaluation_op(
        metrics_path=output_model_path,
        metrics=training.outputs["metrics"]
    )

compiler.Compiler().compile(
    pipeline_func=sentivista,
    package_path="sentivista.json"
)

aiplatform.init(project=PROJECT_ID, location=REGION)

pipeline_job = aiplatform.PipelineJob(
    display_name="sentivista",
    template_path="sentivista.json",
    pipeline_root=PIPELINE_ROOT
)

pipeline_job.run()