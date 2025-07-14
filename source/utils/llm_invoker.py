# modules/llm_invoker.py
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI 
from typing import List, AsyncGenerator
import logging

logger = logging.getLogger("llm_invoker")

async def invoke_llm_streamingly(llm: ChatOpenAI, messages: List[BaseMessage]) -> AsyncGenerator[str, None]:
    try:
        async for token_chunk in llm.astream(messages):
            if hasattr(token_chunk, 'content'):
                yield token_chunk.content
            else: 
                yield str(token_chunk)
    except Exception as e:
        logger.exception(f"Error during LLM stream: {e}")

async def invoke_llm_for_full_response(llm: ChatOpenAI,messages: List[BaseMessage]) -> str:
    try:
        response = await llm.ainvoke(messages)
        if hasattr(response, 'content'):
            return response.content.strip()
        return str(response).strip()
    except Exception as e:
        logger.exception(f"Error calling LLM for full response: {e}")