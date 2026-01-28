def prediction_node(state,markov_model):
    if state["last_item"] is None:
        return state
    state["prediction"] = markov_model.predict_next(state["last_item"])

    return state