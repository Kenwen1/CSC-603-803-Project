import gradio as gr

from brain import answer


def chat(message, history):
    return answer(message, history)


gr.ChatInterface(
    chat,
    title="StudyAI for CSC603/803",
    description=(
        "Your study partner for Generative AI. Ask questions, get concepts "
        "explained, review a topic, or have me quiz you."
    ),
    examples=[
        "Explain how transformers work in simple terms",
        "Quiz me on the topics we covered",
        "Summarize the key ideas about RAG",
        "What's the difference between fine-tuning and prompting?",
    ],
).launch()