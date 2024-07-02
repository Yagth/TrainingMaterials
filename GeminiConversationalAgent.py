from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.agent import AgentFinish

import requests
from pydantic.v1 import BaseModel, Field
import datetime
import wikipediaapi #pip install wikipedia-api
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
class WikipediaInput(BaseModel):
    query: str = Field(..., description="Query to search on Wikipedia")
@tool(args_schema=WikipediaInput)
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a given query."""
    
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(query)
    
    if not page.exists():
        return f"No Wikipedia page found for '{query}'"
    
    return page.summary
tools = [get_current_temperature, search_wikipedia]
functions = [format_tool_to_openai_function(f) for f in tools]

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
).bind(functions=functions)

prompt  = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])

chain = prompt | model | OpenAIFunctionsAgentOutputParser()
response = chain.invoke({"input": "what is the weather is sf?"})
action = response['action']
arguments = response['arguments', {}]
if action == "get_current_temperature":
    second_input = get_current_temperature(**arguments)
elif action == "search_wikipedia":
    second_input = search_wikipedia(**arguments)
else:
    second_input = "Unknown action"
result = chain.invoke({"input": second_input})
print(result)


# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are helpful but sassy assistant"),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])
# chain = prompt | model | OpenAIFunctionsAgentOutputParser()
#
# result1 = chain.invoke({
#     "input": "what is the weather is san fransico located at 37.7749° N, 122.4194° W?",
#     "agent_scratchpad": []
# })
#
# print(result1)
# print(type(result1))
#
# observation = get_current_temperature(result1.tool_input)
# print(observation)

# result2 = chain.invoke({
#     "input": "what is the weather is sf?", 
#     "agent_scratchpad": format_to_openai_functions([(result1, observation)])
# })
# print(result2)
