import streamlit as st
from gtts import gTTS
import whisper
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import tempfile
import os

st.set_page_config(
    page_title="Shees Pod",
)

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Ask user for API key
groq_api_key = st.text_input("Enter your GROQ API Key:", type="password")

if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="qwen-2.5-32b")

# Summarize text function
def summarize_text(text, llm):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following podcast transcript in 3-5 bullet points:\n\n{text}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text)
    return summary

# Main function
def main():
    load_dotenv()
    st.title("EssenceExtractor 🎙")
    st.write("Upload a podcast audio or video file to get a summary.")

    if not groq_api_key:
        st.warning("Please enter your GROQ API key to continue.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "m4a"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("Summarize"):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio.write(uploaded_file.read())
                temp_audio_path = temp_audio.name  # Get the file path

            # Transcribe the uploaded audio file
            with st.spinner("Transcribing audio..."):
                model = load_whisper_model()
                result = model.transcribe(temp_audio_path)
                transcription = result["text"]
                st.success("Transcription complete! ✅")

            # Generate summary using LLM
            with st.spinner("Generating summary..."):
                summary = summarize_text(transcription, llm)
                st.success("Summary generated! 🎉")

            # Display the transcription and summary
            st.subheader("Transcription:")
            st.write(transcription)
            st.sidebar.subheader("Summary:")
            st.sidebar.write(summary)

            # Convert summary text to speech and save to a temporary file
            with st.spinner("Converting summary to speech..."):
                tts = gTTS(text=summary, lang='en')

                # Save to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_speech:
                    speech_path = temp_speech.name
                    tts.save(speech_path)

                st.success("Audio generated! 🔊")

            # Play the generated summary audio
            st.audio(speech_path)

            # Clean up temporary files
            os.remove(temp_audio_path)
            os.remove(speech_path)

if __name__ == "__main__":
    main()
