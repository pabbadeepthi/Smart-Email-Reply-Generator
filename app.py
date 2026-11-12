
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Smart Email Reply Generator")
st.title("Smart Email Reply Generator")
st.write("Generate suggested email replies using DistilGPT2.")

@st.cache_resource
def load_model():
    name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return tokenizer, model

tokenizer, model = load_model()

email = st.text_area("Enter the received email:")
tone = st.selectbox("Reply tone", ["Formal", "Friendly"])
length = st.selectbox("Reply length", ["Short", "Medium"])

max_tokens = 45 if length == "Short" else 80

if st.button("Generate Reply"):
    if not email.strip():
        st.warning("Please enter an email.")
    else:
        prompt = (
            f"Email: {email[:500]}\n"
            f"Write a {tone.lower()} reply to this email.\n"
            "Reply:"
        )

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

        st.subheader("Suggested Reply")
        st.write(reply if reply.strip() else "No reply generated. Please try again.")
