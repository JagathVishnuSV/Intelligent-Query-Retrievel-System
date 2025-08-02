def make_explanation(answer, top_clauses, rationale):
    # Return as structured output per requirements
    from core.models import ClauseInfo, AnswerDetail
    return AnswerDetail(
        answer=answer,
        clauses=[ClauseInfo(section=c["section"], text=c["text"], page=c.get("page")) for c in top_clauses],
        explanation=rationale
    )
