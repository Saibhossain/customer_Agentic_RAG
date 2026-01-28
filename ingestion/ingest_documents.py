import fitz  # PyMuPDF

def ingest_pdf(pdf_path, vector_store):
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            chunks.append(
                f"[Page {page_num + 1}] {text}"
            )

    vector_store.add(chunks)
    return len(chunks)
