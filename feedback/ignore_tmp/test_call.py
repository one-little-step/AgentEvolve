import os
import sys

import requests


API_URL = "https://api.commandcode.ai/provider/v1/chat/completions"
MODEL = "meta/muse-spark-1.2-contributor"


def main():
    # api_key = os.getenv("CMD_API_KEY")
    api_key = "user_4Ndps51xJS1bPJv3uZN4XhsuWJfRrLa7FxfHzVKaqJWWLQCwn49w2daJAVcTYHxR5yHQeU3LLfxV3NB8ZM8aa482"

    if not api_key:
        print("Error: CMD_API_KEY environment variable is not set.")
        print('PowerShell: $env:CMD_API_KEY = "your-new-api-key"')
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Write ma a small haiku about love,and how the perception of love transform from childhood to adulthood, and how it can be both beautiful and painful.",
            }
        ],
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )

        # Raise an exception for HTTP errors such as 401 or 429.
        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]["content"]

        print("Model response:")
        print(message)

    except requests.HTTPError as error:
        print(f"HTTP error: {error}")

        try:
            print("API response:", response.json())
        except ValueError:
            print("API response:", response.text)

        sys.exit(1)

    except requests.RequestException as error:
        print(f"Request failed: {error}")
        sys.exit(1)

    except (KeyError, IndexError, TypeError) as error:
        print(f"Unexpected response structure: {error}")
        print("Full response:", data)
        sys.exit(1)


if __name__ == "__main__":
    main()
