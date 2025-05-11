import requests
import json
import time
import argparse
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()


def test_sentiment_api(url="http://127.0.0.1:5000/predict", model=None, verbose=False):
    """
    Test the sentiment analysis API with various examples
    """
    # Various test cases organized by expected sentiment
    test_cases = {
        "positive": [
            "I love this app, it's amazing!",
            "The weather is beautiful today!",
            "I'm so happy with my new purchase.",
            "This is the best restaurant I've ever been to.",
            "The service was excellent and the staff very friendly.",
            "Congratulations on your promotion, well deserved!",
            "The product exceeded my expectations.",
            "Thank you for your help, you made my day!",
        ],
        "negative": [
            "This is terrible service, would not recommend.",
            "I'm very disappointed with my purchase.",
            "The food was cold and the waiter was rude.",
            "I hate waiting in long lines.",
            "This product is a complete waste of money.",
            "I've never been so frustrated with customer service.",
            "The quality is much worse than advertised.",
            "I regret buying this useless device.",
        ],
        "neutral": [
            "The product works as expected.",
            "I received my order yesterday.",
            "The store opens at 9am.",
            "I'll be there at 5pm.",
            "The meeting is scheduled for tomorrow.",
            "This contains some technical information.",
            "The manual has 100 pages.",
            "The flight departs from terminal B.",
        ],
        "ambiguous": [
            "Well, that was interesting.",
            "I didn't expect that to happen.",
            "This is different from what I'm used to.",
            "That's quite a surprise.",
            "It's certainly unique.",
            "I've never seen anything like it before.",
            "That's an unusual approach.",
            "I wasn't prepared for that.",
        ],
    }

    all_texts = []
    for category, texts in test_cases.items():
        all_texts.extend(texts)

    params = {"texts": all_texts}

    if model:
        params["model"] = model

    start_time = time.time()

    try:
        response = requests.post(url, json=params)
        response_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()

            print(f"{Fore.GREEN}✓ API test successful{Style.RESET_ALL}")
            print(f"Status code: {response.status_code}")
            print(f"Response time: {response_time:.2f} seconds")

            predictions = result.get("predictions", [])
            sentiment_labels = result.get("sentiment_labels", [])

            if verbose:
                print("\nDetailed Results:")
                print(json.dumps(result, indent=4))

            print("\nResults by category:")
            index = 0
            accuracy_by_category = {}

            for category, texts in test_cases.items():
                correct = 0
                print(f"\n{Fore.BLUE}--- {category.upper()} TEXTS ---{Style.RESET_ALL}")

                for text in texts:
                    sentiment = (
                        sentiment_labels[index]
                        if sentiment_labels
                        else (
                            "Positive" if predictions[index] in [1, 4] else "Negative"
                        )
                    )

                    expected_positive = category == "positive"
                    is_positive = sentiment == "Positive"
                    is_correct = (
                        (expected_positive == is_positive)
                        if category in ["positive", "negative"]
                        else None
                    )

                    if is_correct is not None:
                        correct += int(is_correct)

                    if is_correct is None:
                        color = Fore.YELLOW
                    elif is_correct:
                        color = Fore.GREEN
                    else:
                        color = Fore.RED

                    print(f"{text}\n{color}→ {sentiment}{Style.RESET_ALL}")
                    index += 1

                if category in ["positive", "negative"]:
                    accuracy = correct / len(texts) * 100
                    accuracy_by_category[category] = accuracy
                    print(
                        f"{Fore.CYAN}Accuracy for {category}: \
                          {accuracy:.1f}%{Style.RESET_ALL}"
                    )

            if accuracy_by_category:
                total_correct = sum(
                    accuracy_by_category[cat] * len(test_cases[cat])
                    for cat in accuracy_by_category
                )
                total_samples = sum(
                    len(test_cases[cat]) for cat in accuracy_by_category
                )
                overall_accuracy = total_correct / total_samples
                print(
                    f"\n{Fore.GREEN}Overall accuracy: \
                      {overall_accuracy:.1f}%{Style.RESET_ALL}"
                )

        else:
            print(f"{Fore.RED}✗ API test failed{Style.RESET_ALL}")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print(
            f"{Fore.RED}✗ Connection error: Could not connect to the API at \
              {url}{Style.RESET_ALL}"
        )
        print("Make sure the Flask server is running and the URL is correct.")
    except Exception as e:
        print(f"{Fore.RED}✗ Error: {str(e)}{Style.RESET_ALL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the SentiVista sentiment analysis API"
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:5000/predict", help="API endpoint URL"
    )
    parser.add_argument(
        "--model",
        default="sentiment_model_lr.pkl",
        help="Model to use (e.g., nb for Naive Bayes, lr for Logistic Regression)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    print(f"{Fore.CYAN}SentiVista API Test{Style.RESET_ALL}")
    print(f"Testing API at: {args.url}")
    if args.model:
        print(f"Using model: {args.model}")
    print("-" * 50)

    test_sentiment_api(args.url, args.model, args.verbose)
