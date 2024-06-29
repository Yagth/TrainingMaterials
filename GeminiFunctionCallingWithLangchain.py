from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

GOOGLE_API_KEY = 'AIzaSyCCHV3t3O5gDkpHwxHnHQLUMXrvPhFwgqQ'

class pUser(BaseModel):
    name: str
    age: int
    email: str

class Class(BaseModel):
    students: List[pUser]

obj = Class(students=[pUser(name="Jane", age=32, email="jane@gmail.com")])

# print(obj)

class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")

weather_function = convert_pydantic_to_openai_function(WeatherSearch)
# print(weather_function)
 
# The above print statement will output the following:
#
# {'name': 'WeatherSearch',
#  'description': 'Call this with an airport code to get the weather at that airport',
#  'parameters': {
#   'title': 'WeatherSearch',
#   'description': 'Call this with an airport code to get the weather at that airport',
#   'type': 'object',
#   'properties': {
#       'airport_code': {
#           'title': 'Airport Code',
#           'description': 'airport code to get weather for',
#           'type': 'string'
#           }
#       },
#   'required': ['airport_code']
#   }
#}

class WeatherSearch2(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

model_with_function = model.bind(functions=[weather_function])
#
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])

chain = prompt | model_with_function
result = chain.invoke({"input": "what is the weather in sf?"})
print("result 1: ", result)


class ArtistSearch(BaseModel):
     """Call this to get the names of songs by a particular artist"""
     artist_name: str = Field(description="name of artist to look up")
     n: int = Field(description="number of results")

functions = [
     convert_pydantic_to_openai_function(WeatherSearch),
     convert_pydantic_to_openai_function(ArtistSearch),
 ]

model_with_functions = model.bind(functions=functions)
chain = prompt | model_with_function

result = chain.invoke({"input": "what is the weather in sf?"})
# print("Result 2: ", result)

result = chain.invoke({"input": "what are some songs by the beatles?"})
# print("Result 3: ", result)
