"""
Módulo que contiene al LLM basado en Groq usado en el proyecto.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from llm.base import LLM

load_dotenv()


class GroqLLM(LLM):
    """
    Implementación de un LLM usando la API de Groq.
    """

    def __init__(
        self,
        model_name: str = "groq/compound",
        temperature: float = 0.0,
    ) -> None:
        load_dotenv()

        self.client = ChatGroq(
            model=model_name,
            temperature=temperature,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return response.content
