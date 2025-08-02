def answer_question(user_query, top_clauses):
    context = "\n\n".join(f"{c['section']}: {c['text']}" for c in top_clauses)
    # Call LLM with `context` and `user_query` (example with OpenAI GPT-4 API; or local model)
    prompt = f"Given the following policy clauses:\n{context}\n\nAnswer the user's question: {user_query}\nExplain the rationale."
    import openai
    out = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0.1
    )
    return out["choices"][0]["message"]["content"], "Rationale as given by LLM"
