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
    pipeline_root=PIPELINE_ROOT
)
def sentivista():
    ingestion     = data_ingestion_op()
    preprocessing = preprocessing_op(
        input_dataset=ingestion.outputs["dataset"]
    )
    training      = training_op(
        preprocessed_dataset=preprocessing.outputs["preprocessed_dataset"]
    )
    evaluation    = evaluation_op(
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