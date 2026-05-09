import streamlit as st
from transformers import pipeline
from PIL import Image

# function part
 
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    story_generator = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    prompt = f"Write a short story for a 3 to 10-year-old kid about {text}. The story should be in 100 words."
    output = story_generator(prompt, max_new_tokens=120, min_new_tokens=40)[0]['generated_text']
    story_text = output[len(prompt):].strip()
    return story_text

# text2audio
def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story_text)
    return audio_data
  
# main part
# Set up the page
st.set_page_config(page_title="Text to Audio Story", page_icon="🦄")
st.header("🪄Turn Your Text into an Audio Story")
 
uploaded_file = st.file_uploader("📸Upload an image", type=["jpg", "jpeg", "png"])
 
if uploaded_file and st.button("✨Generate Story"):
    image = Image.open(uploaded_file)
    st.image(image, width=300)
 
    st.write("🔍 Looking at your picture...")
    text = img2text(image)
    st.info(f"Scenarios: {text}")
 
    st.write("📝 Writing your story...")
    story = text2story(text)
    st.success(f"Story: {story}")
 
    st.write("🎙️ Recording the story...")
    audio_data = text2audio(story)
    st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
