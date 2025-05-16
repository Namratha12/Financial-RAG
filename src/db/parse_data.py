import json
import pandas as pd
from pathlib import Path


def table_to_markdown(table_ori: list) -> str:
    """
    Convert 'table_ori' (structured table) into Markdown format.
    """
    if not isinstance(table_ori, list) or len(table_ori) < 3:
        return "No valid table data."

    # Use second row as header if first row is not actual headers
    possible_header = table_ori[0]
    if all(cell.strip() == "" for cell in possible_header):
        header = table_ori[1]
        rows = table_ori[2:]
    else:
        header = table_ori[0]
        rows = table_ori[1:]

    # Build markdown table
    markdown = "| " + " | ".join(header) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        markdown += "| " + " | ".join(map(str, row)) + " |\n"

    return markdown.strip()


def load_and_parse_convfinqa(filepath: str) -> pd.DataFrame:
    """
    Load and parse the ConvFinQA dataset.
    """
    with open(filepath, "r") as f:
        raw_data = json.load(f)

    parsed = []
    for i, entry in enumerate(raw_data):
        qa = entry.get("qa", {})
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        doc_id = entry.get("id", f"doc_{i}")

        # Use table_ori for proper tabular structure
        table_ori = entry.get("table_ori", [])
        table_markdown = table_to_markdown(table_ori)

        pre_text = " ".join(entry.get("pre_text", []))
        post_text = " ".join(entry.get("post_text", []))
        context = f"{pre_text} {post_text}".strip()

        parsed.append({
            "id": doc_id,
            "question": question,
            "answer": answer,
            "table_markdown": table_markdown,
            "context": context
        })

    return pd.DataFrame(parsed)


def save_parsed_data(df: pd.DataFrame, out_path: str = "data/parsed_convfinqa.csv"):
    """
    Save the parsed dataframe to CSV format.
    """
    Path("data").mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[✓] Saved parsed data to: {out_path}")


if __name__ == "__main__":
    data_path = "data/train.json"
    if not Path(data_path).exists():
        print(f"[ERROR] File not found: {data_path}")
        exit(1)

    df = load_and_parse_convfinqa(data_path)
    print(df.head(3))  # Preview first 3
    save_parsed_data(df)
