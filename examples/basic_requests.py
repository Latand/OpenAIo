import asyncio

from src.api import OpenAIAPIClient
from src.types import ChatMessage


async def main():
    api_key = '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'
    api = OpenAIAPIClient(api_key)

    system_message = ChatMessage(
        role='system',
        content='You are a chatbot in Telegram chat with a user. You are trying to complete the user\'s message.'
    )
    user_message = ChatMessage(
        role='user',
        content='I want to buy a new car'
    )
    messages = [
        system_message,
        user_message
    ]
    response = await api.request_chat_completion(messages, user_id='123')

    answer = response.choices[0].message.content
    total_tokens = response.usage.total_tokens

    print(f'System: {system_message.content}')
    print(f'User: {user_message.content}')
    print(f'Bot: {answer}')
    print()
    print(f'Total tokens used: {total_tokens}')

    await api.close()

asyncio.run(main())
