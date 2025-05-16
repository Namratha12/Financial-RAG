from langchain.prompts import PromptTemplate

reason_and_answer_prompt_template = PromptTemplate(
    template="""
You are a financial analysis assistant.

<INSTRUCTIONS>
You will receive:
1. A financial QUESTION from a user.
2. TABLE CONTEXT containing structured financial data.
3. NARRATIVE CONTEXT describing additional textual insights.

Your job is to:
- Analyze the TABLE to extract relevant values.
- Use the NARRATIVE for any supporting explanations or footnotes.
- Explain your reasoning step-by-step.
- Be accurate in calculations (e.g., percent change = ((new - old)/old) * 100).
- Avoid answering unless both context types are reasonably relevant.
- Return the final answer inside <ANSWER> tags.

If no useful information is found, respond with “NO ANSWER”.

</INSTRUCTIONS>

<EXAMPLE>
<QUESTION>What was the percentage increase in revenue from 2007 to 2008?</QUESTION>
<TABLE>
Revenue 2007: $9,244.9  
Revenue 2008: $9,362.2
</TABLE>
<CONTEXT>
The report explains that revenues include contributions from newly acquired operations in 2008.
</CONTEXT>
<OUTPUT>
<LOGIC>
Step 1: Revenue 2007 = 9,244.9  
Step 2: Revenue 2008 = 9,362.2  
Step 3: Change = 117.3  
Step 4: Percent Increase = (117.3 / 9,244.9) * 100 = 1.27%
</LOGIC>
<ANSWER>1.27%</ANSWER>
</OUTPUT>
</EXAMPLE>

<QUESTION>{question}</QUESTION>
<TABLE>
{context_table}
</TABLE>
<CONTEXT>
{context_narrative}
</CONTEXT>
""",
    input_variables=["question", "context_table", "context_narrative"],
)


# Prompt 2: Output Scoring

eval_prompt_template = PromptTemplate(
    template="""
<GUIDELINES>
As a response validator, score how closely the SYSTEM RESPONSE matches the REFERENCE ANSWER to the USER QUERY.

Scoring:
- 1 → Identical or extremely close
- 0.5–0.9 → Small margin of error
- 0 → Irrelevant or wrong
Only return a number between 0–1.
</GUIDELINES>

<CASE>
<QUERY>What is the year-over-year percentage change from 2008 to 2009?</QUERY>
<RESPONSE>29.31</RESPONSE>
<REFERENCE>30%</REFERENCE>
<OUTPUT>
0.97
</OUTPUT>
</CASE>

<QUERY>{question}</QUERY>
<RESPONSE>{actual_answer}</RESPONSE>
<REFERENCE>{expected_answer}</REFERENCE>
""",
    input_variables=["question", "actual_answer", "expected_answer"],
)

# Prompt 3: Pull Short Answer
extract_anwer_prompt_template = PromptTemplate(
    template="""
<GUIDELINES>
You’ll receive a detailed explanation for a financial question.
Your job is to extract only the direct final answer.
Return:
- A brief number, value, or phrase
- "NO ANSWER" if missing
</GUIDELINES>

<QUERY>{question}</QUERY>
<EXPLANATION>{generation}</EXPLANATION>
""",
    input_variables=["question", "generation"],
)

# Prompt 4: Document Sifting
filter_context_prompt_template = PromptTemplate(
    template="""
<GUIDELINES>
Given a QUERY and several DOCUMENTS:
- Keep only those containing data relevant to the QUERY
- Summarize only the useful parts
- Return a YAML-style list of sources at the end

DO NOT generate an answer to the query.
</GUIDELINES>

<CASE>
<QUERY>Report 2009 operating cash flow</QUERY>
<DOCS>
<DOC ID=doc1>
2009 operating cash flow was $206,588
</DOC>
<DOC ID=doc2>
Revenue increased 10% in 2009
</DOC>
</DOCS>
<OUTPUT>
2009 operating cash flow was $206,588
sources:
 - doc1
</OUTPUT>
</CASE>

<QUERY>{question}</QUERY>
<DOCS>
{documents}
</DOCS>
""",
    input_variables=["question", "documents"],
)
# Prompt 5: Generate Retrieval Queries

generate_queries_prompt_template = PromptTemplate(
    template="""
<GUIDELINES>
Given a financial QUESTION, draft specific keyword search phrases to surface relevant documents.
Each query should target useful signals.
Return: newline-separated list of search terms only.
</GUIDELINES>

<EXAMPLE>
<QUERY>Show revenue figures over the last 3 quarters</QUERY>
<OUTPUT>
revenue Q1 2024
revenue Q2 2024
revenue Q3 2024
quarterly earnings summary
trailing three quarters revenue
</OUTPUT>
</EXAMPLE>

<QUERY>{question}</QUERY>
""",
    input_variables=["question"],
)