
# TweetNormalizer module for BERTweet
# Based on https://github.com/VinAIResearch/BERTweet

import re
import emoji

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
