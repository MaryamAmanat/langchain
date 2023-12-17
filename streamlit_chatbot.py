# streamlit_chatbot.py

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import langchain

# Load Hugging Face conversational model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Wrap the Hugging Face model using langchain.Model
langchain_model = langchain.Model.from_huggingface(
    model,
    tokenizer=tokenizer,
    input_field="text",
    output_field="logits",
)

def main():
    st.title("Chatbot Interface")

    user_input = st.text_input("You:")
    if user_input:
        # Run the chatbot application
        response_logits = langchain_model(text=user_input)
        response_text = tokenizer.decode(response_logits[0], skip_special_tokens=True)

        # Display the conversation
        st.text(f"You: {user_input}")
        st.text(f"Chatbot: {response_text}")

if __name__ == "__main__":
    main()
