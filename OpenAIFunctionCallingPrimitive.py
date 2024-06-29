import os
import openai
import json

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info``)

# define a function
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
)
# Response has the following format

# {
#   "id": "chatcmpl-9fOawXxFavz5HlXEvP0PN2LaWhsXg",
#   "object": "chat.completion",
#   "created": 1719653018,
#   "model": "gpt-3.5-turbo-0613",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": null,
#         "function_call": {
#           "name": "get_current_weather",
#           "arguments": "{\n  \"location\": \"Boston, MA\"\n}"
#         }
#       },
#       "logprobs": null,
#       "finish_reason": "function_call"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 82,
#     "completion_tokens": 18,
#     "total_tokens": 100
#   },
#   "system_fingerprint": null
# }

response_message = response["choices"][0]["message"]
args = json.loads(response_message["function_call"]["arguments"])

observation = get_current_weather(args)

messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": observation,
        }
)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
)

#The response in the above call will have the following format

# {
#   "id": "chatcmpl-9fOeoNxJORAtQDW8vHifDHpkcGpNr",
#   "object": "chat.completion",
#   "created": 1719653258,
#   "model": "gpt-3.5-turbo-0613",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "The current weather in Boston is 72\u00b0F. It is sunny and windy."
#       },
#       "logprobs": null,
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 77,
#     "completion_tokens": 16,
#     "total_tokens": 93
#   },
#   "system_fingerprint": null
# }
