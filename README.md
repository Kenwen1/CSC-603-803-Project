# StudyAI for CSC603/803

A course-specific chatbot tutor for *CSC 603/803 Generative AI: Fundamentals and Applications*. It uses Retrieval-Augmented Generation (RAG): it searches course PDFs, then has an LLM answer using only what it found, with sources listed under each answer.

## Setup

### 1. Create and activate a virtual environment

From the `studyai` folder. Use Python 3.10 to 3.12 if you can. Very new versions (3.14) may not work with some libraries.

```bash
python3 -m venv venv
```

- Mac/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

### 2. Install the libraries

```bash
pip install pymupdf chromadb sentence-transformers transformers torch gradio accelerate
```

### 3. Build the database (run once, and again whenever materials change)

```bash
python3 ingest.py
```

If you delete or rename a PDF, delete the `db/` folder first, then run `ingest.py` again.

### 4. Start the chatbot

```bash
python3 app.py
```

The first run downloads the Qwen model (several GB), so it can take a while. When you see `Running on local URL`, open `http://127.0.0.1:7860` in your browser. Leave the terminal open while you use the chatbot, and press `Ctrl+C` to stop it.

## Optional: test each part separately

```bash
python3 search.py   # prints the slides found for a sample question
python3 brain.py    # prints a sample answer
```

## Files

| File | What it does |
|---|---|
| `ingest.py` | Reads the PDFs and stores them in the ChromaDB database |
| `search.py` | Finds the pages most relevant to a question |
| `brain.py` | Sends the found pages and the question to the LLM |
| `app.py` | The Gradio web interface |

## Troubleshooting

- **`ModuleNotFoundError`:** the virtual environment isn't activated, or a `pip install` didn't finish.
- **Answers are very slow or memory runs out:** switch to a smaller model, such as `Qwen/Qwen2.5-1.5B-Instruct`, in `brain.py`.