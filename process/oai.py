import openai 
from openai import OpenAI
from Prompts import NEWS_PROMPT
from dotenv import dotenv_values
#method for calling openai 
config = dotenv_values(".env")
client = OpenAI(api_key=config['API_KEY'])
def call_gpt_api(msg):
    response = client.chat.completions.create (
            model="gpt-4",
            messages=msg
        )

    return response.choices[0].message.content