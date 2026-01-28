def document_rag_node(state, vector_store, llm):
    chunks = vector_store.search(state["user_query"], k=5)

    if not chunks:
        state["document_answer"] = (
            "No relevant PDF documents are available to support this answer."
        )
        return state

    context = "\n\n".join(chunks)

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{state["user_query"]}
"""

    state["document_answer"] = llm.generate(prompt)
    return state
