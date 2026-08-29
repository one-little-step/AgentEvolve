import os
import sys

import requests


API_URL = "https://api.commandcode.ai/provider/v1/models"


def main() -> None:
    # api_key = os.getenv("CMD_API_KEY")
    api_key = "user_4Ndps51xJS1bPJv3uZN4XhsuWJfRrLa7FxfHzVKaqJWWLQCwn49w2daJAVcTYHxR5yHQeU3LLfxV3NB8ZM8aa482"

    if not api_key:
        print("Error: CMD_API_KEY environment variable is not set.")
        print('PowerShell: $env:CMD_API_KEY = "your-api-key"')
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        print("Fetching available models...")

        response = requests.get(
            API_URL,
            headers=headers,
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()

        # OpenAI-compatible model-list responses normally contain
        # the array of models under the "data" property.
        models = result.get("data", [])

        if not models:
            print("No models were returned.")
            print("\nFull API response:")
            print(result)
            return

        # Sort models alphabetically by ID.
        models.sort(key=lambda model: model.get("id", "").lower())

        print(f"\nAvailable models: {len(models)}\n")

        for index, model in enumerate(models, start=1):
            model_id = model.get("id", "<unknown>")
            print(f"{index:>3}. {model_id}")

    except requests.Timeout:
        print("Error: The request timed out after 60 seconds.")
        sys.exit(1)

    except requests.ConnectionError as error:
        print(f"Connection error: {error}")
        sys.exit(1)

    except requests.HTTPError as error:
        print(f"HTTP error: {error}")

        try:
            print("API response:")
            print(response.json())
        except ValueError:
            print("API response:")
            print(response.text)

        sys.exit(1)

    except requests.RequestException as error:
        print(f"Request failed: {error}")
        sys.exit(1)

    except ValueError as error:
        print(f"Could not decode the API response as JSON: {error}")
        print("Raw response:")
        print(response.text)
        sys.exit(1)


if __name__ == "__main__":
    main()