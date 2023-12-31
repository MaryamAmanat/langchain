from transformers import AutoModelForCausalLM, AutoTokenizer
import langchain
import streamlit as st

# Load the text generation pipeline
text_generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

# Streamlit code within the script
def streamlit_code():
    st.title("Chatbot Interface")

    user_input = st.text_input("You:")
    if user_input:
        # Generate text based on user input
        response_text = text_generator(user_input, max_length=50, num_return_sequences=1)[0]['generated_text']

        # Display the conversation
        st.text(f"You: {user_input}")
        st.text(f"Chatbot: {response_text}")

# Run the Streamlit code
if __name__ == "__main__":
    streamlit_code()
