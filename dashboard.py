import streamlit as st
import tempfile
from controller.system_controller import SystemController

st.set_page_config(
    page_title="Intelligent Retail AI",
    layout="wide"
)

controller = SystemController()

st.title("🧠 Intelligent Retail Management System")

# -----------------------------
# SIDEBAR – SYSTEM CONTROL
# -----------------------------
st.sidebar.header("⚙️ System Control")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Ingestion",
        "Model Training",
        "Agent Query",
        "Forecast & Recommendation",
        "Logs & System Status"
    ]
)

# -----------------------------
# INGESTION
# -----------------------------
if menu == "Ingestion":
    st.header("📥 Data Ingestion")

    st.subheader("Upload Transaction CSV")
    csv_file = st.file_uploader("Transactions CSV", type=["csv"])

    if csv_file and st.button("Ingest Transactions"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(csv_file.read())
            result = controller.ingest_transactions(tmp.name)
            st.success(result)

    st.subheader("Upload Financial / Policy Documents")
    docs = st.file_uploader(
        "PDF / DOC files",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if docs and st.button("Ingest Documents"):
        paths = []
        for doc in docs:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(doc.read())
            paths.append(tmp.name)
        result = controller.ingest_documents(paths)
        st.success(result)

# -----------------------------
# MODEL TRAINING
# -----------------------------
elif menu == "Model Training":
    st.header("🧮 Train Prediction Model")

    train_csv = st.file_uploader("Training CSV", type=["csv"])

    if train_csv and st.button("Train Model"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(train_csv.read())
            result = controller.train_model(tmp.name)
            st.success(result)

# -----------------------------
# AGENT QUERY
# -----------------------------
elif menu == "Agent Query":
    st.header("🤖 Agentic RAG Query")

    history = st.text_input(
        "Customer Purchase History (comma-separated)",
        "Bread, Milk"
    )

    context = st.text_area(
        "Context / Question",
        "Why is the next item recommended?"
    )

    if st.button("Run Agent"):
        result = controller.query_agent(
            [x.strip() for x in history.split(",")],
            context
        )

        st.subheader("Prediction")
        st.json(result["probabilities"])

        st.subheader("Explanation (LLM)")
        st.write(result["explanation"])

        st.subheader("Evidence (RAG)")
        st.json(result["evidence"])

# -----------------------------
# FORECAST & RECOMMENDATION
# -----------------------------
elif menu == "Forecast & Recommendation":
    st.header("📈 Operational Recommendation")

    traffic = st.slider("Expected Customer Traffic", 0, 500, 120)

    if traffic > 300:
        st.error("🚨 Recommendation: Open 3 cash counters")
    elif traffic > 150:
        st.warning("⚠️ Recommendation: Open 2 cash counters")
    else:
        st.success("✅ Recommendation: Open 1 cash counter")

# -----------------------------
# LOGS & STATUS
# -----------------------------
elif menu == "Logs & System Status":
    st.header("📊 System Status")

    stats = controller.system_stats()
    st.metric("Vectors Stored", stats["vectors"])
    st.metric("Model Trained", stats["model_trained"])

    st.subheader("System Logs")
    try:
        with open("logs/system.log") as f:
            st.text(f.read())
    except:
        st.info("No logs available")
