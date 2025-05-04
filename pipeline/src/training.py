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
    output_component_file="training.yaml"
)
def training_op(
        preprocessed_dataset: Input[Dataset],
        model: Output[Model],
        metrics: Output[Metrics],
        hyperparameters: dict
    ):
    import pandas as pd
    import joblib
    import logging
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # Load preprocessed dataset
    df = pd.read_csv(preprocessed_dataset.path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    x_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    x_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    #Import the model
    model = joblib.load(model.path)
    
    # Initialize the model with hyperparameters
    model = model(**hyperparameters)

    #Train the model
    model.fit(x_train, y_train)
    
    #Make predictions
    predictions = model.predict(x_test)
    
    #Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    df_metrics = pd.DataFrame({
        'Metric': ['MSE', 'R2'],
        'Value': [mse, r2]
    })
    df_metrics.to_csv(metrics.path, index=False)
    
    #Save the model
    joblib.dump(model, model.path)
    
    logging.info(f"Model saved to: {model.path}")
    logging.info(f"Metrics saved to: {metrics.path}")
    logging.info(f"Validation MSE: {mse:.2f}")
    logging.info(f"Validation R2: {r2:.2f}") 
    