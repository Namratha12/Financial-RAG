import csv
import json
import time
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm.auto import tqdm
from langchain_core.messages import HumanMessage

from config import config
from agent import run_agent_pipeline
from prompts import eval_prompt_template
from llm import llm
from utils import format_prompt


def relative_score(a: float, b: float, power: int = 2) -> float:
    return 1 - ((abs(a - b) / max(abs(a), abs(b))) ** power) if a != b else 1.0


def parse_number(val: str) -> float:
    return float(val.replace('%', 'e-2').replace('$', '').replace(',', '').strip())


def compute_accuracy(question: str, predicted: str, expected: str) -> float:
    predicted, expected = predicted.strip().lower(), expected.strip().lower()

    if not predicted and expected:
        return 0.0
    if predicted == expected:
        return 1.0

    try:
        return max(
            relative_score(parse_number(predicted), parse_number(expected)),
            relative_score(parse_number(predicted), float(expected.replace('%', ''))),
            relative_score(float(predicted.replace('%', '')), parse_number(expected))
        )
    except:
        pass

    prompt = eval_prompt_template.format(
        question=question,
        actual_answer=predicted,
        expected_answer=expected,
    )
    response = llm.invoke([HumanMessage(content=format_prompt(prompt))])
    try:
        return float(response.content.strip())
    except:
        return 0.0


def normalize_doc_id(doc_id: str) -> str:
    return doc_id.split("::")[0]  # Strip chunk suffix


def compute_precision(predicted_ids: List[str], expected_id: str) -> float:
    normalized_ids = [normalize_doc_id(pid) for pid in predicted_ids]
    return float(expected_id in normalized_ids) / len(predicted_ids) if predicted_ids else 0.0


def compute_recall(predicted_ids: List[str], expected_id: str) -> float:
    normalized_ids = [normalize_doc_id(pid) for pid in predicted_ids]
    return float(expected_id in normalized_ids)


def load_eval_data(path: str, limit: int = None) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("question", "").strip()]
    return rows[:limit] if limit else rows


def evaluate_one_sample(row: dict) -> dict:
    question = row["question"]
    expected_answer = row["answer"]
    expected_doc_id = row["id"]

    start = time.time()
    result = run_agent_pipeline(question)
    latency = time.time() - start

    predicted_answer = result.get("answer", "")
    generation = result.get("generation", "")
    retrieved_ids = [doc.metadata.get("id", "") for doc in result.get("documents", [])]
    reranked_ids = [doc.metadata.get("id", "") for doc in result.get("reranked_documents", [])]

    accuracy = compute_accuracy(question, predicted_answer, expected_answer)
    ret_precision = compute_precision(retrieved_ids, expected_doc_id)
    ret_recall = compute_recall(retrieved_ids, expected_doc_id)
    rerank_precision = compute_precision(reranked_ids, expected_doc_id)
    rerank_recall = compute_recall(reranked_ids, expected_doc_id)

    print(f"\n[Question] {question}")
    print(f"[Expected Answer] {expected_answer}")
    print(f"[Predicted Answer] {predicted_answer}")
    # print(f"[Retrieved Docs] {retrieved_ids}")
    # print(f"[Reranked Docs] {reranked_ids}")
    # print(f"[Accuracy] {accuracy:.2%}")
    # print(f"[Retrieval Precision] {ret_precision:.2%}")
    # print(f"[Retrieval Recall] {ret_recall:.2%}")
    # print(f"[Rerank Precision] {rerank_precision:.2%}")
    # print(f"[Rerank Recall] {rerank_recall:.2%}")
    # print(f"[Latency] {latency:.2f}s")

    return {
        "id": row["id"],
        "question": question,
        "expected_answer": expected_answer,
        "predicted_answer": predicted_answer,
        "generation": generation,
        "accuracy": accuracy,
        "retrieved_docs": ", ".join(retrieved_ids),
        "retrieval_precision_score": ret_precision,
        "retrieval_recall_score": ret_recall,
        "reranked_docs": ", ".join(reranked_ids),
        "rerank_precision_score": rerank_precision,
        "rerank_recall_score": rerank_recall,
        "latency": latency,
        "prompt": result.get("prompt", "")
    }


def run_evaluation():
    examples = load_eval_data(config.data_path, limit=150)
    results, latencies = [], []

    print(f"[INFO] Evaluating {len(examples)} examples...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(evaluate_one_sample, ex) for ex in examples]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output = future.result()
            results.append(output)
            latencies.append(output["latency"])

    df = pd.DataFrame(results)
    df.to_csv("temp/eval_local.csv", quoting=csv.QUOTE_NONNUMERIC, index=False)

    print("\n[RESULTS SUMMARY]")
    print(f"Average Accuracy: {df['accuracy'].mean():.2%}")
    print(f"Average Retrieval Precision: {df['retrieval_precision_score'].mean():.2%}")
    print(f"Average Retrieval Recall: {df['retrieval_recall_score'].mean():.2%}")
    print(f"Average Rerank Precision: {df['rerank_precision_score'].mean():.2%}")
    print(f"Average Rerank Recall: {df['rerank_recall_score'].mean():.2%}")
    print(f"Average Latency: {df['latency'].mean():.2f}s")
    print("Results saved to eval_local.csv")


if __name__ == "__main__":
    run_evaluation()
