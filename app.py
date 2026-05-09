# Program title: Image to Audio Storytelling App

# Import part
import re

import streamlit as st
from PIL import Image
from transformers import pipeline


# Function part

@st.cache_resource
def load_img2text_model():
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )


@st.cache_resource
def load_story_model():
    return pipeline(
        "text-generation",
        model="pranavpsv/genre-story-generator-v2"
    )


@st.cache_resource
def load_text2audio_model():
    return pipeline(
        "text-to-audio",
        model="Matthijs/mms-tts-eng"
    )


def img2text(image):
    image_to_text_model = load_img2text_model()
    text = image_to_text_model(image)[0]["generated_text"]
    return text.strip()


def text2story(text):
    story_generator = load_story_model()

    prompt = (
        f"Write a 50 to 100 word happy story for a 3 to 10-year-old child. "
        f"The story must be only about this image description: {text}. "
        "Use simple sentences, friendly characters, and a happy ending. "
        "Do not write about war, death, soldiers, scary events, or real people. "
        "Story:"
    )

    output = story_generator(
        prompt,
        max_new_tokens=90,
        min_new_tokens=45,
        do_sample=True,
        temperature=0.6
    )[0]["generated_text"]

    if output.startswith(prompt):
        story_text = output[len(prompt):].strip()
    elif "Story:" in output:
        story_text = output.split("Story:", 1)[1].strip()
    else:
        story_text = output.strip()

    # Basic cleaning
    story_text = story_text.replace("\n", " ").strip()
    story_text = re.sub(r"\s+", " ", story_text)

    # Keep the story within 70 words so the full story can be converted into audio
    words = story_text.split()
    if len(words) > 70:
        first_70_words = " ".join(words[:70])

        last_period = first_70_words.rfind(".")
        last_exclamation = first_70_words.rfind("!")
        last_question = first_70_words.rfind("?")

        last_sentence_end = max(last_period, last_exclamation, last_question)

        if last_sentence_end != -1:
            story_text = first_70_words[:last_sentence_end + 1]
        else:
            story_text = first_70_words + "."

    # Make sure the story ends as a complete sentence
    if story_text and story_text[-1] not in ".!?":
        story_text += "."

    return story_text


def text2audio(story_text):
    audio_pipe = load_text2audio_model()
    audio_data = audio_pipe(story_text)
    return audio_data


# Main part

st.set_page_config(
    page_title="Image to Audio Story",
    page_icon="🦄"
)

st.title("Image to Audio Storytelling App")

st.write(
    "Upload an image. The app will describe the image, generate a short story "
    "for children aged 3 to 10, and convert the story into audio."
)

uploaded_file = st.file_uploader(
    "📸 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and st.button("✨ Generate Story"):
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Looking at your picture..."):
        scenario = img2text(image)

    st.write(f"**Image Description:** {scenario}")

    with st.spinner("📝 Writing your story..."):
        story = text2story(scenario)

    st.write(f"**Generated Story:** {story}")

    word_count = len(re.findall(r"\b[\w']+\b", story))
    st.caption(f"Word count: {word_count}")

    with st.spinner("🎙️ Recording the story..."):
        audio_data = text2audio(story)

    st.audio(
        audio_data["audio"],
        sample_rate=audio_data["sampling_rate"]
    )
