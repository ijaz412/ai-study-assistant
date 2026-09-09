import streamlit as st
import numpy as np
from pathlib import Path
from io import BytesIO

from pypdf import PdfReader
from docx import Document as DocxDocument
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
st.set_page_config(page_title="RAG AI Study Assistant", page_icon="📚", layout="wide")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6
MAX_CONTEXT_CHARS = 18000
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
ALLOWED_MODES = {"auto", "explain", "summarize", "quiz", "flashcards", "compare"}

SYSTEM_PROMPT = """
You are a careful AI study assistant.

Rules:
1. Prefer the student's retrieved study material when answering.
2. Never invent citations, page numbers, or facts from a source.
3. If the retrieved material does not contain enough evidence, clearly say so.
4. Explain difficult concepts at an appropriate student level.
5. For medical topics, distinguish educational information from clinical advice.
6. Be structured, concise, and useful for active learning.
"""

PLANNER_PROMPT = """
Classify the student's request into exactly one mode:
explain, summarize, quiz, flashcards, compare.

Return ONLY the mode name.

Question:
{question}
"""

ANSWER_PROMPT = """
Answer the student's request using the retrieved evidence below when the
question depends on the uploaded study material.

Retrieved evidence:
{context}

Student request:
{question}

Mode:
{mode}

Requirements:
- Make the answer educational and easy to study.
- Mention source document and page when useful.
- If evidence is insufficient, say so instead of guessing.
"""

VERIFY_PROMPT = """
Check whether the draft answer is supported by the retrieved evidence.

Evidence:
{context}

Draft:
{draft}

Return:
PASS
or
FAIL: followed by a short explanation.
"""


# ----------------------------------------------------------------------
# CACHED RESOURCES
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading local embedding model...")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_client():
    api_key = st.secrets.get("GROK_API_KEY", "")
    base_url = st.secrets.get("GROK_BASE_URL", "https://api.groq.com/openai/v1")

    if not api_key:
        return None, base_url

    return OpenAI(api_key=api_key, base_url=base_url), base_url


def get_model_name():
    return st.secrets.get("GROK_MODEL", "llama-3.3-70b-versatile")


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    client, _ = get_client()

    if client is None:
        raise RuntimeError(
            "GROK_API_KEY is missing. Add it in Settings -> Secrets as:\n"
            'GROK_API_KEY = "gsk_your_key_here"'
        )

    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


# ----------------------------------------------------------------------
# DOCUMENT PROCESSING
# ----------------------------------------------------------------------
def extract_text(filename: str, data: bytes):
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": text})
        return pages

    if suffix == ".docx":
        doc = DocxDocument(BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"page": None, "text": text}]

    if suffix == ".txt":
        return [{"page": None, "text": data.decode("utf-8", errors="ignore")}]

    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


def chunk_pages(pages, document_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    counter = 0

    for item in pages:
        text = " ".join(item["text"].split())
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "chunk_id": f"{document_name}-{counter}",
                    "document": document_name,
                    "page": item.get("page"),
                    "text": chunk,
                })
                counter += 1

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks


# ----------------------------------------------------------------------
# IN-MEMORY RAG INDEX (kept in session_state, resets each session)
# ----------------------------------------------------------------------
def init_index():
    if "records" not in st.session_state:
        st.session_state.records = []
    if "matrix" not in st.session_state:
        st.session_state.matrix = None


def add_to_index(chunks):
    if not chunks:
        return

    embedder = load_embedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True)

    st.session_state.records.extend(chunks)

    if st.session_state.matrix is None:
        st.session_state.matrix = np.asarray(vectors)
    else:
        st.session_state.matrix = np.vstack([st.session_state.matrix, vectors])


def search_index(query, k=TOP_K):
    if not st.session_state.records or st.session_state.matrix is None:
        return []

    embedder = load_embedder()
    q = embedder.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(q, st.session_state.matrix)[0]
    indexes = np.argsort(scores)[::-1][:k]

    results = []
    for idx in indexes:
        item = dict(st.session_state.records[int(idx)])
        item["score"] = float(scores[int(idx)])
        results.append(item)

    return results


# ----------------------------------------------------------------------
# WORKFLOW: plan -> retrieve -> answer -> verify
# ----------------------------------------------------------------------
def detect_mode(question, requested_mode):
    if requested_mode != "auto":
        return requested_mode

    result = chat(SYSTEM_PROMPT, PLANNER_PROMPT.format(question=question), temperature=0)
    result = result.strip().lower()

    return result if result in ALLOWED_MODES - {"auto"} else "explain"


def build_context(results):
    parts = []
    total = 0

    for item in results:
        block = (
            f"[Source: {item['document']}; Page: {item.get('page') or 'N/A'}]\n"
            f"{item['text']}\n"
        )
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)

    return "\n".join(parts)


def run_workflow(question, requested_mode="auto"):
    mode = detect_mode(question, requested_mode)

    search_query = question
    if mode in {"quiz", "flashcards"}:
        search_query = f"key facts concepts definitions {question}"

    results = search_index(search_query)
    context = build_context(results)

    if not context:
        return {
            "answer": "I don't have any indexed study material yet. Upload a PDF, DOCX, or TXT file first.",
            "mode": mode,
            "sources": [],
            "verification": "NOT_CHECKED",
        }

    draft = chat(
        SYSTEM_PROMPT,
        ANSWER_PROMPT.format(context=context, question=question, mode=mode),
        temperature=0.2,
    )

    verification = chat(
        SYSTEM_PROMPT,
        VERIFY_PROMPT.format(context=context, draft=draft),
        temperature=0,
    )

    return {
        "answer": draft,
        "mode": mode,
        "sources": results,
        "verification": verification,
    }


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def main():
    init_index()

    st.title("📚 RAG AI Study Assistant")
    st.caption("Upload study material, then ask questions, get summaries, quizzes, or flashcards.")

    client, _ = get_client()
    if client is None:
        st.warning(
            "No API key configured yet. Go to your app's **Settings -> Secrets** "
            "on Streamlit Cloud and add:\n\n"
            '```\nGROK_API_KEY = "gsk_your_key_here"\n'
            'GROK_BASE_URL = "https://api.groq.com/openai/v1"\n'
            'GROK_MODEL = "llama-3.3-70b-versatile"\n```'
        )

    with st.sidebar:
        st.header("Upload material")
        uploaded_files = st.file_uploader(
            "PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for f in uploaded_files:
                suffix = Path(f.name).suffix.lower()
                if suffix not in ALLOWED_EXTENSIONS:
                    st.error(f"Unsupported file type: {f.name}")
                    continue

                already_indexed = any(
                    r["document"] == f.name for r in st.session_state.records
                )
                if already_indexed:
                    continue

                try:
                    data = f.read()
                    pages = extract_text(f.name, data)
                    chunks = chunk_pages(pages, f.name)
                    add_to_index(chunks)
                    st.success(f"Indexed **{f.name}** ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"Could not process {f.name}: {e}")

        st.divider()
        st.metric("Indexed chunks", len(st.session_state.records))

        if st.button("Clear index"):
            st.session_state.records = []
            st.session_state.matrix = None
            st.rerun()

    st.subheader("Ask a question")

    mode = st.selectbox(
        "Mode",
        ["auto", "explain", "summarize", "quiz", "flashcards", "compare"],
        index=0,
    )
    question = st.text_area("Your question or request", height=100)

    if st.button("Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        elif client is None:
            st.error("Add your GROK_API_KEY in Secrets before asking a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    result = run_workflow(question, mode)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    result = None

            if result:
                st.markdown(f"**Mode used:** `{result['mode']}`")
                st.markdown("### Answer")
                st.write(result["answer"])

                st.markdown("### Verification")
                st.write(result["verification"])

                if result["sources"]:
                    with st.expander(f"Sources ({len(result['sources'])})"):
                        for s in result["sources"]:
                            st.markdown(
                                f"- **{s['document']}** "
                                f"(page {s.get('page') or 'N/A'}, "
                                f"score {s['score']:.2f})"
                            )
                            st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))


if __name__ == "__main__":
    main()
