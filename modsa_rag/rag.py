from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from modsa_rag.config import Settings
from modsa_rag.ingest import get_vector_store


SYSTEM_PROMPT = """You are MOD-SA, a KMUTT Student Affairs RAG assistant.

Use only the retrieved context to answer. If the context does not contain enough
information, say that you do not have enough verified information and recommend
contacting the relevant KMUTT office.

Answer in Thai when the question is Thai. Answer in English when the question is
English. Be concise, accurate, and careful with dates, rules, eligibility,
deadlines, fees, scholarships, and registration details.

Retrieved context:
{context}
"""

NO_VERIFIED_INFO_TH = (
    "ขออภัยครับ/ค่ะ ตอนนี้ผมยังไม่พบข้อมูลยืนยันจากฐานความรู้ที่เกี่ยวข้องกับคำถามนี้ "
    "แนะนำให้ตรวจสอบกับหน่วยงานของ KMUTT ที่เกี่ยวข้องโดยตรง"
)

NO_VERIFIED_INFO_EN = (
    "Sorry, I do not have enough verified information in the knowledge base to answer this confidently. "
    "Please confirm with the relevant KMUTT office."
)


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.resolved_llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )


def format_context(docs) -> str:
    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        label = meta.get("title") or meta.get("source", "unknown source")
        page = meta.get("page")
        if page is not None:
            label = f"{label}, page {int(page) + 1}"
        parts.append(f"[{index}] {label}\n{doc.page_content}")
    return "\n\n".join(parts)


def source_summary(docs) -> list[dict[str, object]]:
    seen: set[tuple[object, object]] = set()
    sources: list[dict[str, object]] = []
    for doc in docs:
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        item: dict[str, object] = {"source": source}
        if meta.get("title"):
            item["title"] = meta["title"]
        if meta.get("department"):
            item["department"] = meta["department"]
        if meta.get("source_url"):
            item["url"] = meta["source_url"]
        if page is not None:
            item["page"] = int(page) + 1
        sources.append(item)
    return sources


def is_thai_text(text: str) -> bool:
    return any("\u0E00" <= char <= "\u0E7F" for char in text)


def retrieve_documents(settings: Settings, question: str):
    vector_store = get_vector_store(settings)
    if hasattr(vector_store, "similarity_search"):
        return vector_store.similarity_search(question, k=settings.retrieval_k)

    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retrieval_k})
    return retriever.invoke(question)


def no_verified_info_response(question: str) -> str:
    return NO_VERIFIED_INFO_TH if is_thai_text(question) else NO_VERIFIED_INFO_EN


def answer_question(settings: Settings, question: str) -> dict[str, object]:
    docs = retrieve_documents(settings, question)

    if not docs:
        return {
            "answer": no_verified_info_response(question),
            "sources": [],
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    messages = prompt.format_messages(
        context=format_context(docs),
        question=question,
    )
    response = build_llm(settings).invoke(messages)

    return {
        "answer": response.content,
        "sources": source_summary(docs),
    }
