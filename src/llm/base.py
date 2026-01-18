"""
Módulo que define la clase base para modelos de lenguaje (LLM).
"""

from abc import ABC, abstractmethod


class LLM(ABC):
    """
    Abstracción base para implementar un modelo de lenguaje.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Genera una respuesta a partir de un prompt completo.
        """
        ...
