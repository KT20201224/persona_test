import json
import re


def calculate_persona_generation_metrics(input_data: dict, output_text: str) -> dict:
    """
    회식 토론 페르소나 생성 태스크의 평가 지표를 계산합니다.
    """
    scores = {
        "json_schema_compliance": 0.0,
        "field_coverage": 0.0,
        "classification_accuracy": 0.0,
        "reasoning_depth": 0.0,
        "discussion_readiness": 0.0,
        "specificity": 0.0,
        "consistency": 1.0,
        "extra_text_parsing": 1.0,  # Default to 1.0 if empty
        "overall_score": 0.0,
    }

    parsed_output = {}

    # 1. JSON Schema Compliance
    try:
        parsed_output = json.loads(output_text)
        required_keys = ["persona_prompt", "must_avoid", "preferred", "reasoning"]
        if all(key in parsed_output for key in required_keys):
            if isinstance(parsed_output["must_avoid"], list) and isinstance(
                parsed_output["preferred"], list
            ):
                scores["json_schema_compliance"] = 1.0
            else:
                # partial credit if keys exist but types are wrong? No, strict 0 or 1 for checking types usually.
                # But let's be strict for now.
                scores["json_schema_compliance"] = 0.5  # Penalty for wrong types
    except json.JSONDecodeError:
        scores["json_schema_compliance"] = 0.0
        return scores  # Critical failure, return early

    if scores["json_schema_compliance"] == 0.0:
        return scores

    persona_prompt = parsed_output.get("persona_prompt", "")
    must_avoid = parsed_output.get("must_avoid", [])
    preferred = parsed_output.get("preferred", [])
    reasoning = parsed_output.get("reasoning", "")

    # 2. Field Coverage
    input_fields_check = []
    covered_fields = 0

    if input_data.get("allergies"):
        input_fields_check.append("allergies")
        if any(
            a in str(must_avoid) or a in persona_prompt for a in input_data["allergies"]
        ):
            covered_fields += 1

    if input_data.get("preferred_food_categories"):
        input_fields_check.append("categories")
        if any(
            c in str(preferred) or c in persona_prompt
            for c in input_data["preferred_food_categories"]
        ):
            covered_fields += 1

    if input_data.get("preferred_ingredients"):
        input_fields_check.append("ingredients")
        if any(
            i in str(preferred) or i in persona_prompt
            for i in input_data["preferred_ingredients"]
        ):
            covered_fields += 1

    if input_data.get("extra_text"):
        input_fields_check.append("extra_text")
        reflected = False
        for text in input_data["extra_text"]:
            if (
                text in str(must_avoid)
                or text in str(preferred)
                or text in persona_prompt
            ):
                reflected = True
                break
        if reflected:
            covered_fields += 1

    scores["field_coverage"] = (
        covered_fields / len(input_fields_check) if input_fields_check else 1.0
    )

    # 3. Classification Accuracy
    correct_classifications = 0
    total_classifiable_items = 0

    # Check Allergies -> must_avoid
    for allergy in input_data.get("allergies", []):
        total_classifiable_items += 1
        if any(allergy in item for item in must_avoid):
            correct_classifications += 1

    # Check Preferences -> preferred
    for item in input_data.get("preferred_food_categories", []) + input_data.get(
        "preferred_ingredients", []
    ):
        total_classifiable_items += 1
        if any(item in p_item for p_item in preferred):
            correct_classifications += 1

    scores["classification_accuracy"] = (
        correct_classifications / total_classifiable_items
        if total_classifiable_items > 0
        else 1.0
    )

    # 4. Reasoning Depth
    depth_score = 0.0
    if len(reasoning) >= 50:
        depth_score += 0.3  # Lowered threshold slightly
    if any(
        word in reasoning
        for word in [
            "왜냐하면",
            "따라서",
            "고려하여",
            "때문에",
            "분류",
            "because",
            "due to",
            "considering",
        ]
    ):
        depth_score += 0.3
    if (
        "must_avoid" in reasoning
        or "preferred" in reasoning
        or "제약" in reasoning
        or "선호" in reasoning
    ):
        depth_score += 0.4
    scores["reasoning_depth"] = min(depth_score, 1.0)

    # 5. Discussion Readiness
    readiness_score = 0.0
    if any(
        w in persona_prompt
        for w in ["당신은", "You are", "토론", "discussion", "role", "참여", "역할"]
    ):
        readiness_score += 0.3
    if must_avoid and any(
        w in persona_prompt
        for w in ["절대", "안 됨", "못 먹", "allerg", "never", "cannot"]
    ):
        readiness_score += 0.3
    elif not must_avoid:
        readiness_score += 0.3
    if preferred and any(
        w in persona_prompt
        for w in ["협의", "가능", "선호", "prefer", "negotiab", "조율"]
    ):
        readiness_score += 0.4
    elif not preferred:
        readiness_score += 0.4
    scores["discussion_readiness"] = min(readiness_score, 1.0)

    # 6. Specificity
    spec_score = 0.0
    checks = 0
    hits = 0
    all_keywords = (
        input_data.get("allergies", [])
        + input_data.get("preferred_food_categories", [])
        + input_data.get("preferred_ingredients", [])
    )
    if all_keywords:
        checks += 1
        found = sum(
            1
            for k in all_keywords
            if k in persona_prompt or k in str(must_avoid) or k in str(preferred)
        )
        if found / len(all_keywords) > 0.5:
            hits += 1
    if input_data.get("name"):
        checks += 1
        if input_data["name"] in persona_prompt:
            hits += 1
    scores["specificity"] = hits / checks if checks > 0 else 1.0

    # 7. Consistency (Penalty based)
    penalty = 0.0
    set_avoid = set(must_avoid)
    set_pref = set(preferred)
    if not set_avoid.isdisjoint(set_pref):
        penalty += 0.5
    for allergy in input_data.get("allergies", []):
        if any(allergy in p for p in preferred):
            penalty += 0.5
    scores["consistency"] = max(1.0 - penalty, 0.0)

    # 8. Extra Text Parsing
    extra_texts = input_data.get("extra_text", [])
    if extra_texts:
        parsing_score = 0.0
        strong_pattern = r"(절대|안\s*됨|불가|금지|NO|안돼|never|forbidden)"

        items_checked = 0
        correct_parse = 0

        for text in extra_texts:
            is_strong = re.search(strong_pattern, text)

            # Simple heuristic check
            if is_strong:
                items_checked += 1
                # Check if some parts of the text appear in must_avoid
                # e.g., "매운 음식 절대 안됨" -> split and check
                words = text.split()
                if any(w in str(must_avoid) for w in words if len(w) > 1):
                    correct_parse += 1
            else:
                # Assume preference
                pass

        # If no strong items to check, give full score or adjust
        if items_checked > 0:
            scores["extra_text_parsing"] = correct_parse / items_checked
        else:
            scores["extra_text_parsing"] = 1.0  # No strong items to parse
    else:
        scores["extra_text_parsing"] = 1.0

    scores["overall_score"] = sum(scores.values()) / len(scores)
    return scores
