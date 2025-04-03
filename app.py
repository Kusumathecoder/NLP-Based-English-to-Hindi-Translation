import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM  # ✅ Add TFAutoModelForSeq2SeqLM


from datasets import load_dataset
import os

# Streamlit App Title
st.title("🌍 English to Hindi Translator")

# Model and Tokenizer Caching
@st.cache_resource
def load_model():
    model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = load_model()

# User Input
input_text = st.text_area("Enter English text to translate:", "My name is Bappy. My YouTube channel is DSwithBappy.")

if st.button("Translate"):
    if input_text.strip():
        # Tokenization & Translation
        tokenized = tokenizer([input_text], return_tensors='np')
        output = model.generate(**tokenized, max_length=128)
        
        # Decode translation
        with tokenizer.as_target_tokenizer():
            translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Display Translation
        st.success("Translated text (Hindi):")
        st.write(f"👉 {translated_text}")

    else:
        st.warning("⚠️ Please enter some text to translate.")

# Footer
st.markdown("---")
st.markdown("🔗 **Powered by Helsinki-NLP/opus-mt-en-hi** | 🛠️ Built with Streamlit & Transformers")
