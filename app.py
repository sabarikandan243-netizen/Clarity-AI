import streamlit as st
import clarity_ai_v33 as ai

# Page configuration
st.set_page_config(page_title="Clarity AI")

st.title("CLARITY AI")
st.write("AI Hallucination Verifier")

# --- Part 1: File Upload ---
st.write("### 1. Add your Documents (Optional)")
uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            ai.add_document(uploaded_file.name, text)
            st.write("Document added successfully. You can now verify statements based on it.")
        elif uploaded_file.name.endswith(".pdf"):
            st.write("PDF file uploaded. System will use it for verification.")
    except Exception as e:
        st.write("Error loading file: " + str(e))

# --- Part 2: Claim Verification ---
st.write("### 2. Verify a Statement")
claim = st.text_input("Type a statement here (e.g., 'Chennai is the capital of Tamil Nadu.')")

if st.button("Check Statement"):
    if claim:
        with st.spinner("Checking sources..."):
            result = ai.clarity_ai(claim)
            
            status = result.get("result", "UNKNOWN")
            
            if status == "SUPPORTED":
                st.write("Result: SUPPORTED")
                st.write("This statement is true.")
            elif status == "REFUTED":
                st.write("Result: REFUTED")
                st.write("This statement is false.")
            else:
                st.write("Result: NEEDS REVIEW")
                st.write("Not enough evidence to confirm.")

            st.write("### Evidence:")
            evidences = result.get("supporting_evidence", [])
            if evidences:
                for ev in evidences:
                    st.write("- " + ev['text'])
            else:
                st.write("No strong evidence found.")
    else:
        st.write("Please type a statement first.")

# --- Part 3: Feedback Form Link ---
st.write("---")
st.write("### User Feedback")
st.markdown("[Click here to give your feedback](https://docs.google.com/forms/d/e/1FAIpQLSe87wEQIgChoc_5BNHEZe_JqfJlOEKT7vGz18P6kypDtMgTlQ/viewform?usp=header)")
