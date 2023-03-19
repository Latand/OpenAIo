from typing import List

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: ChatMessage


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 1
    top_p: float = 1
    user: str = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int = None
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    choices: List[Choice]
    usage: Usage


class Embedding(BaseModel):
    embedding: List[float]
    index: int
    object: str


class EmbeddingResponse(BaseModel):
    model: str
    object: str
    data: list[Embedding]
    usage: Usage


class CompletionResult(BaseModel):
    text: str
    parse_mode: str = None
    original: str
