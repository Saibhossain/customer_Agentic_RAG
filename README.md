# 🛒 Agentic RAG for Customer Behavior Analysis (Prototype)

A **lightweight, production-style prototype** demonstrating how a **RAG Agent** can be used for **customer behavior analysis** — specifically, predicting the **probability of the next item purchase** and **explaining it using retrieved historical evidence**.

This project is inspired by **retail scenarios like Walmart**, but implemented using **modern AI concepts** such as:
- Retrieval-Augmented Generation (RAG)
- Agentic decision-making
- Probabilistic sequence modeling

> ⚠️ This is a **prototype / proof-of-concept**, not a full-scale recommender system.

---

## 🎯 Project Objective

Given a customer’s recent purchase history:
1. Predict **what item the customer is likely to buy next**
2. Compute **probabilities** of next-item choices
3. **Explain** the prediction by retrieving similar historical purchase sequences using a **RAG Agent**

---

## 🧠 Key Idea

Traditional recommendation systems *predict* but rarely *explain*.

This project introduces a **RAG Agent** that:
- Decides **when retrieval is needed**
- Retrieves similar historical purchase sequences
- Grounds predictions in **real transaction evidence**

Example output:
```json
{
  "last_item": "Milk",
  "next_item_probabilities": {
    "Eggs": 1.0
  },
  "retrieved_examples": [
    "Bread-> Milk-> Eggs",
    "Diapers-> Wipes-> Baby Lotion",
    "Soap-> Shampoo"
  ]
}
```

## System Architecture (Conceptual)
    User Purchase History
            ↓
       RAG Agent
            ├── Decide: Should I retrieve?
            ├── Retrieve similar purchase sequences (Vector DB)
            ├── Predict next-item probabilities
            └── Generate grounded explanation

## Project Structure

    customer_Agentic_RAG/
    ├── data/
    │   └── transactions.csv          # Sample transaction data
    ├── embeddings/
    │   └── vector_store.py            # FAISS-based vector retrieval
    ├── models/
    │   └── transition_model.py        # Probabilistic transition model
    ├── agent/
    │   └── rag_agent.py               # Agentic RAG logic
    ├── app.py                         # Main application
    ├── README.md  
    └── requirements.txt


## Dummy dataset used 

    customer_id,timestamp,item
    C1,2026-01-01 10:00,Bread
    C1,2026-01-01 10:05,Milk
    C1,2026-01-01 10:10,Eggs

## Install dependencies

    pip install -r requirements.txt


### Disclaimer

This project is a conceptual prototype meant for educational and experimental purposes.
It is not optimized for large-scale production deployment.

### Author

MD Saib Hossain
[Email](saibhossain5@gmail.com) | [Linkedin](https://www.linkedin.com/in/saib-hossain-182834229/)

Prototype project on Agentic RAG for Customer Behavior Analysis