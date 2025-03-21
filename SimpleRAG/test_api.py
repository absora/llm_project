from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())


class ChatClient:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=os.getenv(api_key),
            base_url=os.getenv(base_url)
        )

    def chat(self, model, message):
        response = self.client.chat.completions.create(
            model=model,
            messages=message
        )
        return response.choices[0].message


openai_client = ChatClient(api_key="OPENAI_API_KEY", base_url="OPENAI_BASE_URL")
zju_client = ChatClient(api_key="ZJU_API_KEY", base_url="ZJU_BASE_URL")

openai_response = openai_client.chat(
    model="gpt-3.5-turbo",
    message=[
        {
            "role": "user",
            "content": "Who are you",
        }
    ]
)
print(openai_response)

zju_response = zju_client.chat(
    model="deepseek-r1-distill-qwen",
    message=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "3.11与3.9谁大"
        }
    ]
)
print(zju_response)

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_BASE_URL")
# )
#
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {
#             "role": "user",
#             "content": "Who are you",
#         }
#     ]
# )
#
# print(response.choices[0].message)

# zju_client = OpenAI(
#     api_key=os.getenv("ZJU_API_KEY"),
#     base_url=os.getenv("ZJU_BASE_URL")
# )
#
# response = zju_client.chat.completions.create(
#     model="deepseek-r1-distill-qwen",
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant."
#         },
#         {
#             "role": "user",
#             "content": "3.11与3.9谁大"
#         }
#     ]
# )
#
# print(response.choices[0].message)
