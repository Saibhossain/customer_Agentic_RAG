import streamlit as st
import pandas as pd
import tempfile

from langgraph_app import agent  # your compiled LangGraph agent
from embeddings.vector_store import VectorStore
from ingestion.ingest_documents import ingest_pdf
from models.markov_model import MarkovNextItemModel

st.set_page_config(
    page_title=" Smart Retail Intelligence",
    layout="wide",
)

st.title(" Smart Retail Intelligence Dashboard")
st.caption("Agentic RAG • Customer Behavior • Explainable AI")

@st.cache_data
def load_data():
    df = pd.read_csv("data/update_dataset11.csv")
    df["item"] = df["item"].str.lower().str.strip()
    return df

df = load_data()
st.sidebar.header(" Customer Control")

customer_ids = sorted(df["customer_id"].unique())
selected_customer = st.sidebar.selectbox(
    "Select Customer",
    customer_ids
)

customer_df = df[df["customer_id"] == selected_customer]

ordered_items = (
    customer_df
    .sort_values(["timestamp", "transaction_id", "item_sequence"])
    ["item"]
    .tolist()
)

last_item = ordered_items[-1]

st.sidebar.success(f"Last item: **{last_item}**")
st.sidebar.markdown("---")
st.sidebar.subheader(" Knowledge Upload (PDF)")

doc_vector_store = VectorStore()

uploaded_pdf = st.sidebar.file_uploader(
    "Upload sales / policy PDF",
    type=["pdf"]
)

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    chunks = ingest_pdf(pdf_path, doc_vector_store)
    st.sidebar.success(f"PDF ingested ({chunks} chunks)")


tab1, tab2, tab3, tab4 = st.tabs(
    [" Customer Data", " Prediction", " Agent Explanation", " Agent Graph"]
)

with tab1:
    st.subheader(f"Customer {selected_customer} – Purchase History")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.dataframe(
            customer_df.sort_values(
                ["timestamp", "transaction_id", "item_sequence"]
            ),
            use_container_width=True
        )

    with col2:
        st.metric("Loyalty Level", customer_df["loyalty_level"].iloc[0])
        st.metric("Transactions", customer_df["transaction_id"].nunique())
        st.metric("Total Items", len(customer_df))

with tab2:
    st.subheader("Next Product Prediction")

    st.markdown(f"**Last purchased item:** `{last_item}`")
    model = MarkovNextItemModel()
    model.fit(df)

    predictions = model.predict_next(last_item)

    if predictions:
        pred_df = pd.DataFrame(
            predictions.items(),
            columns=["Next Item", "Probability"]
        )

        st.bar_chart(
            pred_df.set_index("Next Item"),
            use_container_width=True
        )

        st.dataframe(pred_df, use_container_width=True)

    else:
        st.warning("No prediction available for this item.")

with tab3:
    st.subheader("Agentic AI Explanation")

    user_question = st.text_input(
        "Ask a question (prediction, reason, document insight):",
        value="What will this customer buy next and why?"
    )

    if st.button("Ask Agent"):
        with st.spinner("Agent reasoning..."):
            result = agent.invoke({
                "user_query": user_question,
                "last_item": last_item
            })

        st.markdown("### Final Answer")
        st.success(result["final_answer"])

with tab4:
    st.subheader("Agentic RAG Execution Graph")

    png_bytes = agent.get_graph().draw_mermaid_png()
    st.image(png_bytes, caption="LangGraph Agent Planner")

    st.markdown("""
                **Explanation:**
                - Planner decides which tools to use
                - Prediction node runs Markov model
                - Document node retrieves PDF context
                - Generator produces final answer
                """)
