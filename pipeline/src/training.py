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
        model_checkpoint: str,
        preprocessed_dataset: Input[Dataset],
        metrics: Output[Metrics],
        new_model: Output[Model],
    ):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd
    import joblib
    import logging
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    def tokenize_function(examples):
        return tokenizer(examples['cleaned_text'], truncation=True, padding='max_length', max_length=128)

       # Enhanced metrics calculation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, pos_label=1),
            'recall': recall_score(labels, predictions, pos_label=1),
            'f1': f1_score(labels, predictions, pos_label=1)
        }
        
        return metrics

    #Load preprocessed dataset
    df = pd.read_csv(preprocessed_dataset.path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    #Tokenize the dataset
    train_df = train_df.map(tokenize_function, batched=True)
    test_df = test_df.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_df.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_df.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    #Define model path
    model_path = model_checkpoint
    
    #Import the model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=3,  # Increased from 1 to 3 for better performance
        per_device_train_batch_size=16,  # Reduced for larger dataset to avoid memory issues
        per_device_eval_batch_size=32,
        evaluation_strategy='steps',  # Evaluate at steps rather than just epochs for better monitoring
        eval_steps=1000,  # Evaluate every 1000 steps
        save_strategy='steps',
        save_steps=1000,
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',  # Use accuracy instead of loss
        save_total_limit=2,  # Keep more checkpoints
        report_to='none',  # Disable wandb reporting
        # Add learning rate scheduling
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,  # Add weight decay for regularization
        gradient_accumulation_steps=4,  # Accumulate gradients to effectively increase batch size
        fp16=True  # Use mixed precision training if GPU supports it
    )

    # Create Trainer with enhanced configuration
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_df,
        eval_dataset=test_df,
        compute_metrics=compute_metrics
    )

    #Train the model
    trainer.train()
    
    #Metrics
    mtrcs = trainer.evaluate()
    mtrcs.to_csv(metrics.path, index=False)
    
    #Save the model
    model.save_pretrained(new_model.path)
    
    logging.info(f"Model saved to: {new_model.path}")
    logging.info(f"Metrics saved to: {metrics.path}")
    