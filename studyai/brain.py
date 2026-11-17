import re
import time

from transformers import pipeline

from search import find_pieces

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",  # try "meta-llama/Llama-3.2-3B-Instruct" later
    device_map="auto",
    dtype="auto",  # if your transformers version complains, change to torch_dtype="auto"
)

MAX_HISTORY = 8  # how many past messages the tutor remembers

SYSTEM_PROMPT = """You are StudyAI, a friendly, conversational study partner for CSC603/803 (Generative AI: Fundamentals and Applications).

You can explain concepts, answer questions, summarize topics, quiz the student, grade their answers, make flashcards or practice problems, compare ideas, and give study tips.

How to use the course materials:
- Excerpts from the course PDFs are provided as "Course material excerpts", labeled with [file, page]. Treat them as the source of truth for what this course says, and mention the file and page when you rely on them.
- If the excerpts don't cover the question, you may still help using your general knowledge, but say so clearly, for example: "This isn't in the course materials, but generally...".
- Never make up what the course said.

How to behave:
- Be conversational, warm, and concise. Remember what was said earlier in the conversation.
- When quizzing: ask ONE question at a time and wait for the student's answer. Then say whether it was right, explain why, and offer the next question.
- If the student asks for a quiz or review without a topic, ask which topic they want, or suggest a few from the excerpts.
- If the student seems confused, explain more simply and use an example."""

SHORT_PROMPT = """ You are StudyAI, a friendly, conversational study partner for CSC603/803 (Generative AI: Fundamentals and Applications).
Answer rules:
- Put the direct answer in the first sentence. For simple questions, answer in 4-7 sentences total.
- Do not use headings or "Explanation"/"Summary" sections. Do not restate or describe the excerpts.
- Only add background, examples, or extra detail if the student asks for more, or asks to summarize, review, or be quizzed.
- Treat the course excerpts as the source of truth and mention the file and page only when you rely on them. If they don't cover the question, answer from general knowledge in one sentence starting with "This isn't in the course materials, but generally...". Never make up what the course said.

How to behave:
- Be conversational, warm, and concise. Remember what was said earlier in the conversation.
- When quizzing: ask ONE question at a time and wait for the student's answer. Then say whether it was right, explain why, and offer the next question.
- If the student asks for a quiz or review without a topic, ask which topic they want, or suggest a few from the excerpts.
- If the student seems confused, explain more simply and use an example."""

SMALL_TALK = re.compile(
    r"^(hi|hello|hey|thanks|thank you|thx|ok|okay|cool|great|bye|got it)\b[\s!.?]*$",
    re.IGNORECASE,
)

# Requests that need more material than a single question does
BROAD_WORDS = ("quiz", "test me", "summar", "review", "flashcard", "practice", "overview", "explain", "detail")


def text_of(content):
    """Gradio versions differ: content may be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def clean_history(history):
    """Turn Gradio's history into a simple list of role/content messages."""
    messages = []
    for item in history:
        if isinstance(item, dict):
            messages.append(
                {"role": item["role"], "content": text_of(item["content"])}
            )
        else:  # older tuple format: (user, bot)
            user, bot = item
            if user:
                messages.append({"role": "user", "content": text_of(user)})
            if bot:
                messages.append({"role": "assistant", "content": text_of(bot)})
    for m in messages:  # drop old source lists to save space
        m["content"] = m["content"].split("\n\n**Related course pages:**")[0]
    return messages[-MAX_HISTORY:]

def standalone_simple_question(message, history):
    if not history or len(message.split()) > 8:
        return message
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    return f"{last_user} {message}"

def standalone_question(message, history):
    """Rewrite a follow-up like 'why?' into a full search query using the chat so far."""
    if not history:
        return message
    convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in history[-4:])
    prompt = [
        {
            "role": "system",
            "content": (
                "Rewrite the student's last message as one standalone search query "
                "for finding relevant course slides. Use the conversation for context. "
                "Output only the query."
            ),
        },
        {"role": "user", "content": f"Conversation:\n{convo}\n\nLast message: {message}"},
    ]
    out = generator(prompt, max_new_tokens=60, do_sample=False)
    return out[0]["generated_text"][-1]["content"].strip() or message


def answer(message, history=None):
    history = clean_history(history or [])

    is_broad = any(w in message.lower() for w in BROAD_WORDS)

    pieces = []
    if not SMALL_TALK.match(message.strip()):
        if not is_broad:
            query = standalone_simple_question(message, history)
        else:
            query = standalone_question(message, history)
        how_many = 8 if is_broad else 5
        pieces = find_pieces(query, how_many=how_many)

    context = "\n\n".join(
        f'[{p["file"]}, page {p["page"]}]\n{p["text"]}' for p in pieces
    )

    style = "" if is_broad else "\n\n(Answer directly in 4-7 sentences. No Headings, very little to no background.)"

    if context:
        user_turn = f"Course material excerpts:\n{context}\n\nStudent message: {message}{style}"
    else:
        user_turn = message + style

    system_prompt = SYSTEM_PROMPT if is_broad else SHORT_PROMPT
    messages = [{"role": "system", "content": system_prompt}] + history
    messages.append({"role": "user", "content": user_turn})

    result = generator(messages, max_new_tokens=500 if is_broad else 250, do_sample=True, temperature=0.4)
    reply = result[0]["generated_text"][-1]["content"]

    if pieces:
        pages = sorted({(p["file"], p["page"]) for p in pieces})
        sources = [f"{f} (page {n})" for f, n in pages]
        reply += "\n\n**Related course pages:**\n- " + "\n- ".join(sources)
    return reply


if __name__ == "__main__":
    print(answer("What is a transformer?"))


#for time benchmarking
# def answer(message, history=None):
#     total_start = time.perf_counter()
#     history = clean_history(history or [])

#     rewrite_time = 0.0
#     retrieval_time = 0.0
#     generation_time = 0.0

#     pieces = []

#     if not SMALL_TALK.match(message.strip()):
#         # Measure query rewriting
#         start = time.perf_counter()
#         query = standalone_question(message, history)
#         rewrite_time = time.perf_counter() - start

#         # Measure ChromaDB retrieval
#         start = time.perf_counter()
#         how_many = 4 if any(
#             w in message.lower() for w in BROAD_WORDS
#         ) else 3

#         pieces = find_pieces(query, how_many=how_many)
#         retrieval_time = time.perf_counter() - start

#     context = "\n\n".join(
#         f'[{p["file"]}, page {p["page"]}]\n{p["text"]}'
#         for p in pieces
#     )

#     if context:
#         user_turn = (
#             f"Course material excerpts:\n{context}"
#             f"\n\nStudent message: {message}"
#         )
#     else:
#         user_turn = message

#     messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
#     messages.append({"role": "user", "content": user_turn})

#     # Measure Qwen generation
#     start = time.perf_counter()

#     result = generator(
#         messages,
#         max_new_tokens=500,
#         do_sample=True,
#         temperature=0.4
#     )

#     generation_time = time.perf_counter() - start
#     reply = result[0]["generated_text"][-1]["content"]

#     if pieces:
#         pages = sorted({(p["file"], p["page"]) for p in pieces})
#         sources = [f"{f} (page {n})" for f, n in pages]
#         reply += "\n\n**Related course pages:**\n- " + "\n- ".join(sources)

#     total_time = time.perf_counter() - total_start

#     # Display performance results in the chatbot
#     reply += (
#         "\n\n---\n"
#         "**Performance Metrics**\n"
#         f"- Query rewriting: {rewrite_time:.2f}s\n"
#         f"- Document retrieval: {retrieval_time:.2f}s\n"
#         f"- LLM generation: {generation_time:.2f}s\n"
#         f"- Total processing: {total_time:.2f}s"
#     )

#     return reply

