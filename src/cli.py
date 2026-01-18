"""
Módulo que contiene la interfaz de línea de comandos para interactuar con el agente RAG.
"""


def interactive_loop(agent):
    """
    Función que inicia un bucle interactivo para que el usuario pueda hacer preguntas al agente RAG.

    :param agent: Agente RAG con el que interactuar.
    :type agent: RAGAgent
    """
    print("Causalito listo. Escribe tu pregunta o 'exit' para salir.")
    while True:
        question = input("\n>> ").strip()
        if question.lower() in {"exit", "salir"}:
            print("Hasta luego.")
            break
        try:
            answer, sources = agent.answer(question)
            print("\n" + "-" * 40 + "\n[Respuesta]\n" + answer + "\n" + "-" * 40)
            if sources:
                print(f"\nFuentes: {sources}")
            print("-" * 40)
        except Exception as e:
            print(f"[ERROR] {e}")
