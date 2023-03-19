# OpenAIo
This is a very primitive OpenAI Chat API wrapper for async usage.

There are only 2 methods are added now:
- request_chat_completion - for getting [chat completion](https://platform.openai.com/docs/api-reference/chat)
- get_embeddings - for getting [embeddings](https://platform.openai.com/docs/api-reference/embeddings), that can be used for similarity search within messages history

Works with Python 3.10