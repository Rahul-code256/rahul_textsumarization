import streamlit as st
from groq import Groq

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="PragyanAI - Text Summarization",
    page_icon="🦜",
    layout="centered"
)

# -----------------------------
# Logo (Optional)
# -----------------------------
try:
    st.image("PragyanAI_Transperent.png", use_container_width=True)
except:
    pass

st.title("🦜 PragyanAI - Text Summarization App")
st.divider()

# -----------------------------
# Groq Client
# -----------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# -----------------------------
# Function to Generate Summary
# -----------------------------
def generate_summary(text):

    prompt = f"""
You are an expert text summarizer.

Summarize the following text clearly and accurately.

Text:
{text}

Summary:
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# Text Input
# -----------------------------
txt_input = st.text_area(
    "Enter your text",
    height=250,
    placeholder="Paste your paragraph here..."
)

# -----------------------------
# Form
# -----------------------------
with st.form("summary_form"):

    submitted = st.form_submit_button("Generate Summary")

    if submitted:

        if txt_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            with st.spinner("Generating summary..."):

                try:
                    summary = generate_summary(txt_input)

                    st.success("Summary Generated Successfully!")

                    st.subheader("Summary")

                    st.write(summary)

                except Exception as e:
                    st.error(f"Error: {e}")
