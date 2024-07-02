from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.agent import AgentFinish

import requests
import wikipedia
import wikipediaapi
from pydantic.v1 import BaseModel, Field
import datetime

GOOGLE_API_KEY = 'AIzaSyCCHV3t3O5gDkpHwxHnHQLUMXrvPhFwgqQ'


# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'

@tool
def get_wikipedia_summary(title: str) -> str:
    """Fetch a summary of the given Wikipedia title."""
    try:
        # Provide a custom user-agent to avoid issues with Wikipedia API
        # wiki_wiki = wikipediaapi.Wikipedia('en', user_agent="LangchainAssistant/1.0")
        page = wikipedia.page(title)
        if not page.exists():
            return "Page does not exist"
        return page.summary
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for the title: {title}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation page found. Please be more specific: {e.options}"

# Add the tools
tools = [get_current_temperature, get_wikipedia_summary]
functions = [format_tool_to_openai_function(f) for f in tools]

# Initialize the model with updated functions
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
).bind(functions=functions)

# Define the chain with updated functions
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful but sassy assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
chain = prompt | model | OpenAIFunctionsAgentOutputParser()

# Function to process the LLM's response and invoke the correct tool
def process_and_invoke(response):
    if isinstance(response, AgentFinish):
        print("Response from LLM:", response.return_values['output'])
        return response.return_values['output']
    else:
        tool_name = response.tool
        tool_input = response.tool_input
        
        if tool_name == 'get_current_temperature':
            observation = get_current_temperature(tool_input.latitude, tool_input.longitude)
        elif tool_name == 'get_wikipedia_summary':
            observation = get_wikipedia_summary(tool_input)
        else:
            raise ValueError(f"Unknown tool name: {tool_name}")

    # Pass the result back to the LLM
    result = chain.invoke({
        "input": f"The result of the function is: {observation}",
        "agent_scratchpad": format_to_openai_functions([(response, observation)])
    })
    
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        return result.get('output', '')

# Example usage for Wikipedia search
result1 = chain.invoke({
    "input": "Tell me about the Golden Gate Bridge",
    "agent_scratchpad": []
})

# Manually call the identified function and get the result
observation = process_and_invoke(result1)
print(observation)

# Example usage for current temperature
# result2 = chain.invoke({
#     "input": "What is the weather in San Francisco located at 37.7749° N, 122.4194° W?",
#     "agent_scratchpad": []
# })

# # Manually call the identified function and get the result
# observation_temp = process_and_invoke(result2)
# print(observation_temp)
