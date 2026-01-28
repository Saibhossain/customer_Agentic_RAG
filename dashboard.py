import streamlit as st
import pandas as pd
from models.markov_model import MarkovNextItemModel
from agents.explanation_agent import ExplanationAgent
from llm.ollama_client import OllamaClient
import tempfile
from ingestion.ingest_documents import ingest_pdf
from embeddings.vector_store import VectorStore
from agents.document_rag_nod import DocumentSummaryAgent
from llm.ollama_client import OllamaClient

st.markdown("---")
st.subheader(" PDF Knowledge Base (RAG)")

uploaded_pdf = st.file_uploader(
    "Upload a PDF report",
    type=["pdf"]
)

# initialize doc vector store
doc_vs = VectorStore()

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    chunks = ingest_pdf(pdf_path, doc_vs)
    st.success(f"PDF ingested successfully ({chunks} chunks).")

# RAG Query
question = st.text_input(
    "Ask a question from uploaded PDF"
)

if question:
    doc_agent = DocumentSummaryAgent(
        doc_vs,
        OllamaClient(model="gemma3:1b")
    )

    answer = doc_agent.explain_from_pdf(question)
    st.markdown("### RAG Answer")
    st.markdown(answer)


explanation_agent = ExplanationAgent(
    OllamaClient(model="gemma3:1b")
)

st.set_page_config(
    page_title="Customer Purchase Prediction Dashboard",
    layout="wide"
)

st.title("🛒 Customer Purchase Prediction Dashboard")
st.caption("Markov-based Next Item Prediction")


@st.cache_data
def load_data():
    df = pd.read_csv("/Users/mdsaibhossain/code/python/customer_Agentic_RAG/data/update_dataset11.csv")
    df["item"] = df["item"].str.strip().str.lower()
    return df

df = load_data()

@st.cache_resource
def train_model(_df):
    model = MarkovNextItemModel()
    model.fit(_df)
    return model

model = train_model(df)
st.sidebar.header("Customer Selection")

customers = sorted(df["customer_id"].unique())
selected_customer = st.sidebar.selectbox("Select Customer ID", customers )

customer_df = df[df["customer_id"] == selected_customer]
st.subheader(f"Customer: {selected_customer}")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Purchase History")
    st.dataframe(
        customer_df.sort_values(
            ["timestamp", "transaction_id", "item_sequence"]
        ),
        use_container_width=True
    )

with col2:
    st.markdown("###  Customer Profile")

    loyalty = customer_df["loyalty_level"].iloc[0]
    total_transactions = customer_df["transaction_id"].nunique()
    total_items = len(customer_df)

    st.metric("Loyalty Level", loyalty)
    st.metric("Total Transactions", total_transactions)
    st.metric("Total Items Purchased", total_items)


ordered_items = (
    customer_df
    .sort_values(["timestamp", "transaction_id", "item_sequence"])
    ["item"]
    .tolist()
)

last_item = ordered_items[-1]


st.markdown("---")
st.subheader(" Next Product Prediction")
st.markdown(f"**Last purchased item:** `{last_item}`")
predictions = model.predict_next(last_item, top_k=5)

if predictions:
    pred_df = pd.DataFrame(
        predictions.items(),
        columns=["Next Item", "Probability"]
    )

    st.bar_chart( pred_df.set_index("Next Item"), use_container_width=True )

    st.dataframe(pred_df, use_container_width=True)

else:
    st.warning("No prediction available. Not enough historical transitions for this item." )

with st.expander("LLM Explanation (Gemma)"):
    explanation_text = explanation_agent.explain_markov_prediction(
        customer_id=selected_customer,
        last_item=last_item,
        predictions=predictions
    )
    st.markdown(explanation_text)
