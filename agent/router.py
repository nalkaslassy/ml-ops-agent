"""
Router for selecting the right ML model from a plain-English problem description.

This module supports two routing modes:
1) LLM routing via Anthropic (optional, if package + API key are present)
2) Deterministic keyword fallback (always available, no API key required)

The fallback guarantees routing still works when external API usage is blocked,
misconfigured, or rate/credit limited.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from typing import Callable, Dict, List

MODEL_NAMES = ("ChurnModel", "SentimentModel", "FraudModel", "SegmentationModel")

MODEL_TO_IMPORT: Dict[str, str] = {
    "ChurnModel": "churn.churn_model:ChurnModel",
    "SentimentModel": "sentiment.sentiment_model:SentimentModel",
    "FraudModel": "fraud.fraud_model:FraudModel",
    "SegmentationModel": "segmentation.segmentation_model:SegmentationModel",
}

DATA_NEEDED: Dict[str, List[str]] = {
    "ChurnModel": [
        "Structured customer dataset with numeric/categorical features.",
        "Binary churn label column (0 = retained, 1 = churned).",
        "Optional ID columns listed in drop_cols so they are excluded from training.",
    ],
    "SentimentModel": [
        "Dataset with a text column (reviews/comments/feedback).",
        "Binary label column (0/1 or positive/negative).",
        "Consistent language/text normalization in your source data.",
    ],
    "FraudModel": [
        "Structured transaction dataset with numeric/categorical features.",
        "Binary fraud label column (0 = legitimate, 1 = fraud).",
        "Historically imbalanced data where fraud is a small minority.",
    ],
    "SegmentationModel": [
        "Structured dataset with numeric feature columns.",
        "No target label is required.",
        "Optional ID columns listed in drop_cols so only feature columns are clustered.",
    ],
}

KEYWORDS: Dict[str, List[str]] = {
    "ChurnModel": [
        "churn",
        "cancel",
        "cancellation",
        "leave",
        "retention",
        "retain",
        "attrition",
        "unsubscribe",
        "renewal",
        "not renewing",
        "customer loss",
    ],
    "SentimentModel": [
        "sentiment",
        "review",
        "reviews",
        "positive",
        "negative",
        "feedback",
        "comment",
        "comments",
        "opinion",
        "tone",
        "nlp",
        "text classification",
    ],
    "FraudModel": [
        "fraud",
        "fraudulent",
        "suspicious",
        "scam",
        "anomaly",
        "illegitimate",
        "chargeback",
        "transaction risk",
        "payment fraud",
        "rare event",
        "rare events",
    ],
    "SegmentationModel": [
        "segment",
        "segmentation",
        "cluster",
        "clustering",
        "group",
        "grouping",
        "cohort",
        "persona",
        "customer types",
        "natural groups",
        "unsupervised",
    ],
}

SYSTEM_PROMPT = """You are a routing agent for an ML platform.
Given a business problem, select exactly one model from:
- ChurnModel
- SentimentModel
- FraudModel
- SegmentationModel

Call the route_model tool with your selection.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "enum": list(MODEL_NAMES),
        },
        "reason": {"type": "string"},
        "data_needed": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["model", "reason", "data_needed"],
    "additionalProperties": False,
}


def _normalize(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _keyword_route(problem: str) -> dict:
    text = _normalize(problem)
    scores = {name: 0 for name in MODEL_NAMES}

    for model_name, words in KEYWORDS.items():
        for word in words:
            if word in text:
                # Longer phrases are usually more specific, so reward them more.
                scores[model_name] += 2 if " " in word else 1

    # Tie-breaking heuristics for common ambiguous verbs.
    if "predict" in text and any(w in text for w in ("leave", "cancel", "churn", "renewal")):
        scores["ChurnModel"] += 2
    if any(w in text for w in ("text", "review", "feedback", "comment")):
        scores["SentimentModel"] += 1
    if "transaction" in text and any(w in text for w in ("fraud", "suspicious", "scam", "anomaly")):
        scores["FraudModel"] += 2
    if any(w in text for w in ("group", "cluster", "segment", "cohort")) and "predict" not in text:
        scores["SegmentationModel"] += 2

    best_model = max(scores, key=scores.get)
    if scores[best_model] == 0:
        # Default to churn for generic "at-risk customer" style business prompts.
        best_model = "ChurnModel"

    reason = {
        "ChurnModel": "This looks like a customer retention/cancellation prediction problem, which matches churn modeling.",
        "SentimentModel": "This looks like text opinion classification (positive/negative), which matches sentiment analysis.",
        "FraudModel": "This looks like rare suspicious-event detection in transactional data, which matches fraud modeling.",
        "SegmentationModel": "This looks like unsupervised grouping of similar customers/entities, which matches segmentation.",
    }[best_model]

    return {
        "model": best_model,
        "reason": reason,
        "data_needed": DATA_NEEDED[best_model],
    }


def _anthropic_route(problem: str) -> dict:
    # Imported lazily so the module still works without anthropic installed.
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    routing_tool = {
        "name": "route_model",
        "description": "Select the appropriate ML model for the given business problem.",
        "input_schema": OUTPUT_SCHEMA,
    }
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        tools=[routing_tool],
        tool_choice={"type": "tool", "name": "route_model"},
        messages=[{"role": "user", "content": problem.strip()}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    parsed = tool_block.input
    model_name = parsed.get("model")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unexpected model returned by LLM: {model_name}")
    if not parsed.get("data_needed"):
        parsed["data_needed"] = DATA_NEEDED[model_name]
    return parsed


def route(problem: str, prefer_llm: bool = True) -> dict:
    """
    Route a plain-English business problem to one model.

    Args:
        problem: Plain-English description of the task.
        prefer_llm: If True, try Anthropic first (when configured), otherwise use fallback.

    Returns:
        dict with keys: model, reason, data_needed
    """
    if not problem or not problem.strip():
        raise ValueError("Problem description cannot be empty.")

    if prefer_llm:
        try:
            if os.getenv("ANTHROPIC_API_KEY"):
                return _anthropic_route(problem)
        except Exception:
            # Silent fallback is intentional so local routing always remains usable.
            pass

    return _keyword_route(problem)


def get_model_class(model_name: str) -> Callable:
    """
    Resolve and return the model class object from the routed model name.
    """
    if model_name not in MODEL_TO_IMPORT:
        raise ValueError(
            f"Unknown model '{model_name}'. Must be one of: {', '.join(MODEL_NAMES)}"
        )

    module_path, class_name = MODEL_TO_IMPORT[model_name].split(":", maxsplit=1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ImportError(f"Could not import '{class_name}' from '{module_path}'.")
    return model_class


def route_and_get_class(problem: str, prefer_llm: bool = True) -> dict:
    """
    Convenience helper that routes and also returns the class object.

    Returns:
        {
            "model": str,
            "reason": str,
            "data_needed": list[str],
            "model_class": class
        }
    """
    result = route(problem=problem, prefer_llm=prefer_llm)
    result["model_class"] = get_model_class(result["model"])
    return result


if __name__ == "__main__":
    tests = [
        "Which customers are likely to cancel in the next 30 days?",
        "Classify these app-store reviews into positive or negative.",
        "Detect suspicious card transactions where fraud is rare.",
        "Group customers into natural personas for targeting.",
    ]

    print("Router self-test")
    print("=" * 40)
    for item in tests:
        result = route(item)
        print(f"\nProblem: {item}")
        print(f"Model: {result['model']}")
        print(f"Reason: {result['reason']}")
        print("Data needed:")
        for need in result["data_needed"]:
            print(f"- {need}")