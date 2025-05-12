import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.flask.app import clean_tweet


def test_clean_tweet():
    # Test URL removal
    assert "hello world" in clean_tweet("hello world https://example.com")
    
    # Test mention removal
    assert "@user" not in clean_tweet("hello @user")
    
    # Test special character removal
    assert "!" not in clean_tweet("hello world!")
    
    # Test number removal
    assert "123" not in clean_tweet("hello123")
    
    # Test multiple spaces
    assert "hello    world" not in clean_tweet("hello    world")
    assert "hello world" in clean_tweet("hello    world")
