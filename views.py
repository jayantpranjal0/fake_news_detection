from process.oai import call_gpt_api
from Prompts import NEWS_PROMPT
#method for calling openai 
def ask_gpt(news_article,news_class):
    context =f'Use the given {news_article} and the {news_class}'
    messages=[
                {"role": "system", "content": NEWS_PROMPT},
                {"role": "user", "content": context}
            ]
    return call_gpt_api(messages)




