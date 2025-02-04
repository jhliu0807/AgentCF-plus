import dashscope
import time
from openai import OpenAI

client = OpenAI(
    api_key = ""
)

def get_response_from_dashscope(prompt, api_key, model):
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    response = dashscope.Generation.call(
        api_key=api_key,
        model=model, 
        messages=messages,
        result_format='message'
    )
    while response['status_code'] != 200:
        time.sleep(3)
        response = dashscope.Generation.call(
            api_key=api_key,
            model=model, 
            messages=messages,
            result_format='message'
        )
    responseText = response['output']['choices'][0]['message']['content']  # Response text
    return responseText

def get_response_from_openai(prompt, model):
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    max_retries = 10  # Maximum number of retries
    retries = 0
    while retries < max_retries:
        try:
            # Use the Completion API to generate text
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=800,  # Maximum number of tokens, adjustable as needed
                n=1,  # Number of responses to generate, usually 1
                stop=None,  # Stop condition, can be a specific string or None
                temperature=0.7,  # Creativity parameter, between 0 and 1; higher is more creative
            )
            # Check if the response is valid
            if len(response.choices) > 0:
                responseText = response.choices[0].message.content.strip()  # Response text
                return responseText
        except Exception as e:
            print(f"Request failed: {e}, retrying... (Attempt: {retries + 1})")
            retries += 1
            time.sleep(2)  # Wait for 2 seconds before retrying

    print("Reached the maximum number of retries, exiting the program.")
    return None  # Return None if the maximum retries are reached