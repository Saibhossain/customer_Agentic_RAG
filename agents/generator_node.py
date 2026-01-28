def generator_node(state, llm):
    last_item = state.get("last_item")
    prediction = state.get("prediction")
    document_answer = state.get("document_answer")

    prompt = f"""
            You are an AI analytics assistant.
            
            You ALREADY have the customer's context.
            
            Customer last purchased item:
            {last_item}
            
            Predicted next-item probabilities (from a Markov model):
            {prediction}
            
            Additional document-based context:
            {document_answer}
            
            User question:
            {state["user_query"]}
            
            Your task:
            - Do NOT ask for more information
            - Explain the prediction clearly
            - Assume the customer context is complete
            - Give a final answer in simple business language
            """

    state["final_answer"] = llm.generate(prompt)
    return state
