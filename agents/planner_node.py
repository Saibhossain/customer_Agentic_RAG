def planner_node(state, llm):
    prompt = f"""
            You are an AI planner.
        
            Customer context is available:
            - last_item = {state.get("last_item")}
        
            User query:
            {state["user_query"]}
        
            Decide which tools are needed.
        
            Tools:
            - PREDICTION
            - DOCUMENT
            - PREDICTION+DOCUMENT
        
            Return ONLY one of the above.
            """

    decision = llm.generate(prompt).strip().upper()
    state["plan"] = decision
    return state
