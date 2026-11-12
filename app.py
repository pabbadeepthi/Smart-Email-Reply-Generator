
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Smart Email Reply Generator", page_icon="📧")

st.title("Smart Email Reply Generator")
st.write("Generate email replies using AI.")

@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Loading AI model..."):
    tokenizer, model = load_model()

email = st.text_area("Enter the email you received:")

tone = st.selectbox("Choose reply tone:", ["Formal", "Friendly"])

if st.button("Generate Reply"):
    if not email.strip():
        st.warning("Please enter an email first.")
    else:
        prompt = f"Write a {tone.lower()} email reply to this message:\n{email}\nReply:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        reply = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("Generated Reply")
        st.write(reply[len(prompt):].strip() or reply)
