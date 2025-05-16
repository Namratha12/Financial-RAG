import os
import re
import time
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cohere
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from config import config
from llm import llm
from prompts import (
    reason_and_answer_prompt_template,
    extract_anwer_prompt_template,
    filter_context_prompt_template,
    generate_queries_prompt_template,
)
from state import AgentState
from vector_store import vector_store, RelevantDocumentRetriever
from utils import format_prompt

# Initialize clients
cohere_client = cohere.Client(os.environ.get("COHERE_API_KEY"))
fallback_document_lookup = RelevantDocumentRetriever(config.data_path)

def extract_question(state: AgentState) -> AgentState:
    return state.model_copy(update={"question": state.messages[-1].content})

def generate_queries(state: AgentState, _) -> AgentState:
    prompt = generate_queries_prompt_template.format(question=state.question)
    response = llm.invoke([HumanMessage(content=format_prompt(prompt))])
    queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    if state.question not in queries:
        queries.append(state.question)
    return state.model_copy(update={"queries": queries})

def extract_years(text: str) -> list[str]:
    return re.findall(r"\b(19\d{2}|20\d{2})\b", text)

def retrieve_documents(state: AgentState, _) -> AgentState:
    if config.use_ground_truth_retrieval:
        return retrieve_from_ground_truth(state)
    return retrieve_from_vector_search(state)


def retrieve_from_ground_truth(state: AgentState) -> AgentState:
    documents = fallback_document_lookup.query(state.question)
    return state.copy(update={"documents": documents})


def retrieve_from_vector_search(state: AgentState) -> AgentState:
    seen_ids = set()
    results = []

    def search(query):
        return vector_store.similarity_search(query, k=config.top_k_retrieval)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(search, q) for q in state.queries]
        for future in as_completed(futures):
            for doc in future.result():
                doc_id = doc.metadata.get("id")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    results.append(doc)

    # filter by years in question
    years = extract_years(state.question)
    if years:
        filtered = [doc for doc in results if any(year in doc.metadata.get("id", "") for year in years)]
        if filtered:
            results = filtered

    return state.model_copy(update={"documents": results})

def rerank_documents(state: AgentState, _) -> AgentState:
    if config.use_ground_truth_retrieval:
        table, narrative = split_context(state.documents)
        return state.copy(update={
            "reranked_documents": state.documents,
            "context_table": table,
            "context_narrative": narrative
        })

    docs = [{"text": doc.page_content, "id": doc.metadata["id"]} for doc in state.documents]
    response = cohere_client.rerank(
        model=config.reranker_model_name,
        query=state.question,
        documents=docs,
        top_n=config.top_k_rerank,
    )
    reranked = [state.documents[result.index] for result in response.results]
    table, narrative = split_context(reranked)

    return state.model_copy(update={
        "reranked_documents": reranked,
        "context_table": table,
        "context_narrative": narrative
    })



def filter_context_documents(state: AgentState, _) -> AgentState:
    prompt = filter_context_prompt_template.format(
        question=state.question,
        documents=format_documents(state.reranked_documents)
    )
    response = llm.invoke([HumanMessage(content=format_prompt(prompt))])
    raw = response.content.replace("<OUTPUT>", "").replace("</OUTPUT>", "")

    try:
        context, sources = re.split("sources:", raw, flags=re.IGNORECASE, maxsplit=1)
        context = context.strip()
        source_list = [s.strip().lstrip("-") for s in sources.strip().split("\n") if s.strip()]
    except ValueError:
        context = raw.strip()
        source_list = []

    return state.copy(update={"context": context, "sources": source_list})


def generate_answer(state: AgentState, _) -> AgentState:
    prompt = reason_and_answer_prompt_template.format(
        question=state.question,
        context_table=state.context_table,
        context_narrative=state.context_narrative,
    )
    if config.disable_llm_generation:
        return state.copy(update={"prompt": prompt, "generation": "[GENERATION DISABLED]"})
    response = llm.invoke([HumanMessage(content=format_prompt(prompt))])
    return state.copy(update={"prompt": prompt, "generation": response.content})


def extract_final_answer(state: AgentState) -> AgentState:
    if config.disable_llm_generation:
        return state.copy(update={"answer": "NO ANSWER"})

    match = re.search(r"<ANSWER>(.*?)</ANSWER>", state.generation, re.DOTALL)
    if match:
        return state.copy(update={"answer": match.group(1).strip()})

    fallback_prompt = extract_anwer_prompt_template.format(
        question=state.question,
        generation=state.generation
    )
    fallback = llm.invoke([HumanMessage(content=format_prompt(fallback_prompt))])
    return state.copy(update={"answer": fallback.content.strip()})


#Helper Functions

def split_context(docs: List[Document]) -> Tuple[str, str]:
    table, narrative = [], []
    for doc in docs:
        content = doc.page_content
        table_part, narrative_part = "", ""

        if "passage:" in content:
            stripped = content.split("passage:", 1)[-1]
            if "\n\n" in stripped:
                table_part, narrative_part = stripped.split("\n\n", 1)
            else:
                table_part = stripped
        else:
            table_part = content

        table.append(table_part.strip())
        narrative.append(narrative_part.strip())

    return "\n\n".join(table), "\n\n".join(narrative)


def format_documents(docs: List[Document]) -> str:
    return "\n".join(
        f"<DOC ID={doc.metadata.get('id', idx)}>\n{doc.page_content}\n</DOC>"
        for idx, doc in enumerate(docs)
    )


# Entry Point

def run_agent_pipeline(question: str) -> dict:
    state = AgentState(messages=[HumanMessage(content=question)])
    state = extract_question(state)
    state = generate_queries(state, config)
    state = retrieve_documents(state, config)
    state = rerank_documents(state, config)
    state = filter_context_documents(state, config)
    state = generate_answer(state, config)
    state = extract_final_answer(state)

    return {
        "question": state.question,
        "answer": state.answer,
        "generation": state.generation,
        "documents": state.documents,
        "reranked_documents": state.reranked_documents,
        "prompt": state.prompt,
    }