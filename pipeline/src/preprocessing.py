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
from src.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE,
    output_component_file="preprocessing.yaml"
)
def preprocessing_op(    
        input_dataset: Input[Dataset],
        preprocessed_dataset: Output[Dataset]
    ):
    import pandas as pd
    import logging
    import re
    import emoji
    
    # Normalize tweets with improved function to handle errors
    def safe_normalize(text):
        try:
            if pd.isna(text) or text == '':
                return ''
            return normalizeTweet(str(text))
        except Exception as e:
            print(f"Error normalizing text: {str(e)}")
            return str(text)  # Return original text if normalization fails

    def normalizeTweet(tweet):
        """
        Normalize tweet text:
        1. Replace URLs with HTTPURL
        2. Replace user mentions with @USER
        3. Replace emojis with text
        4. Other normalizations for Twitter-specific content
        """
        # Replace URLs with HTTPURL
        tweet = re.sub(r'https?://\S+', 'HTTPURL', tweet)
        
        # Replace user mentions with @USER
        tweet = re.sub(r'@\w+', '@USER', tweet)
        
        # Replace emojis with text representation
        tweet = emoji.demojize(tweet)
        
        # Other normalizations
        tweet = tweet.replace('#', ' #')  # Add space before hashtags
        tweet = re.sub(r'\s+', ' ', tweet)  # Replace multiple spaces with single space
        
        return tweet.strip()

    seed = 42
    df = pd.read_csv(input_dataset.path)

    df['normalized_text'] = df['text'].apply(safe_normalize)
    df = df[df['normalized_text'].str.len() > 5]
    
    # Rebalance if filtering removed any tweets
    if df['target'].value_counts().size > 1:  # Check that both classes still exist
        min_class_size = min(df['target'].value_counts())
        df_positive = df[df['target'] == 1].sample(min_class_size, random_state=seed)
        df_negative = df[df['target'] == 0].sample(min_class_size, random_state=seed)
        df = pd.concat([df_positive, df_negative]).reset_index(drop=True)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    df.to_csv(preprocessed_dataset.path, index=False)
    logging.info(f"Preprocessed dataset saved to: {preprocessed_dataset.path}")