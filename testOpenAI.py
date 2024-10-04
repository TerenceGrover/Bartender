import openai
import os

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")  # or replace with your API key like: openai.api_key = "your-api-key"

def test_openai_api():
    try:
        # Make a simple API request to check if it's working
        response = openai.Completion.create(
            model="text-davinci-003",  # You can use "gpt-3.5-turbo" as well
            prompt="Say hello world!",
            max_tokens=5
        )
        print("API Response:", response.choices[0].text.strip())
    
    except openai.error.AuthenticationError:
        print("Error: Invalid API Key")
    except openai.error.APIConnectionError:
        print("Error: Failed to connect to OpenAI API")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_openai_api()
