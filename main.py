# main.py

import argparse
from langchain_core.messages import HumanMessage

from config import config  # Singleton config instance
from state import AgentState
from agent import (
    extract_question,
    generate_queries,
    retrieve_documents,
    rerank_documents,
    filter_context_documents,
    generate_answer,
    extract_final_answer,
)


def run_agent_pipeline(question: str):
    print("\nRunning Agent Pipeline\n")

    # Initialize agent state
    state = AgentState(messages=[HumanMessage(content=question)])

    # Run pipeline steps
    state = extract_question(state)
    state = generate_queries(state, config)
    state = retrieve_documents(state, config)
    state = rerank_documents(state, config)
    state = filter_context_documents(state, config)
    state = generate_answer(state, config)
    state = extract_final_answer(state)

    # Output
    print("\nAnswer:")
    print(state.answer)
    print("\nReasoning:")
    print(state.generation)

    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ConvFinQA RAG pipeline.")
    parser.add_argument("--question", type=str, required=True, help="Financial question to answer")
    args = parser.parse_args()

    run_agent_pipeline(args.question)
