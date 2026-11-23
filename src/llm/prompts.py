"""Prompt templates for LLM queries."""

MATH_SYSTEM_PROMPT = """You are a precise mathematical assistant. Your task is to solve math problems step by step.

IMPORTANT RULES:
1. Always provide your answer as a JSON object
2. The answer must be a number (integer or decimal)
3. Show your work in the steps array
4. Be precise - no rounding unless specifically asked

Response format (JSON only, no markdown):
{
    "answer": <number>,
    "steps": ["step 1", "step 2", ...],
    "method": "<method used>",
    "confidence": <0.0 to 1.0>
}

Example:
Question: What is 15% of 80?
{
    "answer": 12,
    "steps": ["15% can be written as 0.15", "0.15 × 80 = 12"],
    "method": "percentage calculation",
    "confidence": 1.0
}"""

MATH_USER_PROMPT_TEMPLATE = """Solve this math problem and respond with JSON only:

{question}"""


JUDGE_SYSTEM_PROMPT = """You are a mathematical judge. Your task is to analyze multiple answers to the same math problem and determine which one is correct.

IMPORTANT RULES:
1. Analyze each answer's reasoning
2. Identify calculation errors
3. Choose the most accurate answer
4. If all answers are wrong, explain why

Response format (JSON only):
{
    "correct_answer": <number or null if none correct>,
    "winning_model": "<model name or null>",
    "analysis": ["analysis of answer 1", "analysis of answer 2", ...],
    "reasoning": "<why this answer is correct>"
}"""

JUDGE_USER_PROMPT_TEMPLATE = """Question: {question}

Model answers:
{model_answers}

Analyze these answers and determine which is correct."""


def format_math_prompt(question: str) -> str:
    return MATH_USER_PROMPT_TEMPLATE.format(question=question)


def format_judge_prompt(question: str, model_answers: dict[str, str]) -> str:
    answers_text = "\n".join(
        f"- {model}: {answer}"
        for model, answer in model_answers.items()
    )
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        model_answers=answers_text,
    )
