from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.agent import AgentFinish

import requests
from pydantic import BaseModel, Field
import datetime

GOOGLE_API_KEY = 'AIzaSyCCHV3t3O5gDkpHwxHnHQLUMXrvPhFwgqQ'

class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

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
    
    return {'temperature': current_temperature}

class WikipediaInput(BaseModel):
    query: str = Field(..., description="Query string to search on Wikipedia")

@tool(args_schema=WikipediaInput)
def search_wikipedia(query: str) -> dict:
    """Search Wikipedia for the given query."""
    
    BASE_URL = "https://en.wikipedia.org/w/api.php"
    
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': query
    }

    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        search_results = results['query']['search']
        if search_results:
            return {'snippet': search_results[0]['snippet']}  # Return the snippet of the first search result
        else:
            return {'snippet': "No results found"}
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

tools = [get_current_temperature, search_wikipedia]
functions = [format_tool_to_openai_function(f) for f in tools]

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
).bind(functions=functions)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful but sassy assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
chain = prompt | model | OpenAIFunctionsAgentOutputParser()

result1 = chain.invoke({
    "input": "What is the weather in San Francisco located at 37.7749° N, 122.4194° W?",
    "agent_scratchpad": []
})

observation = get_current_temperature(**result1.tool_input)

result2 = chain.invoke({
    "input": "What is the weather in SF?", 
    "agent_scratchpad": format_to_openai_functions([(result1, observation)])
})
print(result2)

result3 = chain.invoke({
    "input": "Tell me about Python programming language",
    "agent_scratchpad": []
})

observation2 = search_wikipedia(**result3.tool_input)

result4 = chain.invoke({
    "input": "Tell me about Python programming language", 
    "agent_scratchpad": format_to_openai_functions([(result3, observation2)])
})
print(result4)
