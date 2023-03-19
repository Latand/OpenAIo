from src.base import BaseClient
from src.types import ChatMessage, ChatCompletionResponse, ChatCompletionRequest, EmbeddingResponse


class OpenAIAPIClient(BaseClient):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self._api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        base_url = "https://api.openai.com"
        super().__init__(base_url)

    async def request_chat_completion(
            self, messages: list[ChatMessage], user_id=None, model=None
    ) -> ChatCompletionResponse:
        request = ChatCompletionRequest(
            model=model or self.model, messages=messages, user=user_id
        )
        status, result = await self._make_request(
            method="POST",
            url="/v1/chat/completions",
            json=request.dict(),
            headers=self.headers,
        )
        return ChatCompletionResponse(**result)

    async def get_embeddings(self, text: str) -> EmbeddingResponse:
        text = await clean_text_for_embedding(text)
        status, result = await self._make_request(
            method="POST",
            url="/v1/embeddings",
            json={"input": text, "model": "text-embedding-ada-002"},
            headers=self.headers,
        )
        return EmbeddingResponse(**result)


async def clean_text_for_embedding(text: str) -> str:
    characters_to_remove = ["\n", "\t", "\r"]
    for char in characters_to_remove:
        text = text.replace(char, " ")
    return text
