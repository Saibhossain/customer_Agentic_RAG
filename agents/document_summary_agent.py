
class DocumentSummaryAgent:
    def __init__(self, vector_store, llm):
        self.vs = vector_store
        self.llm = llm

    def explain_from_pdf(self, question):
        retrieved_chunks = self.vs.search(question, k=5)

        if not retrieved_chunks:
            return "No relevant information found in uploaded documents."

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
                    You are a business analytics assistant.
                    
                    Answer the following question using ONLY the context below.
                    
                    Context:
                    {context}
                    
                    Question:
                    {question}
                    
                    Give a clear, business-friendly explanation.
                    """

        return self.llm.generate(prompt)
