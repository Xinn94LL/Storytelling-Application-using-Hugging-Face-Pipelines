import re

import streamlit as st
from PIL import Image
from transformers import pipeline


MIN_WORDS = 50
MAX_WORDS = 100

UNSAFE_STORY_WORDS = (
    "death",
    "dead",
    "died",
    "tragic",
    "orphan",
    "killed",
    "war",
    "injury",
    "injuries",
    "blood",
    "scary",
    "afraid",
    "taunt",
    "violent",
    "abandoned",
    "orphanage",
)

GENERIC_CAPTION_WORDS = (
    "image",
    "picture",
    "photo",
    "illustration",
    "scene",
    "there",
    "with",
    "standing",
    "sitting",
)


@st.cache_resource
def load_img2text_model():
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )


@st.cache_resource
def load_story_model():
    return pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
    )


@st.cache_resource
def load_text2audio_model():
    return pipeline(
        "text-to-audio",
        model="Matthijs/mms-tts-eng",
    )


def img2text(image):
    image_to_text_model = load_img2text_model()
    result = image_to_text_model(image)
    return result[0]["generated_text"].strip()


def build_story_prompt(caption):
    return (
        f"Image description: {caption}\n\n"
        "Write one cheerful story for children aged 3 to 10. "
        "The story must be based only on the image description. "
        "Use 50 to 100 words, simple words, and a happy ending. "
        "Return only the story."
    )


def count_words(text):
    return len(re.findall(r"\b[\w']+\b", text))


def trim_to_word_limit(text):
    words = text.split()
    if len(words) <= MAX_WORDS:
        return text

    trimmed = " ".join(words[:MAX_WORDS])
    sentence_end = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
    if sentence_end > 0 and count_words(trimmed[:sentence_end]) >= MIN_WORDS:
        trimmed = trimmed[: sentence_end + 1]
    elif not trimmed.endswith((".", "!", "?")):
        trimmed += "."

    return trimmed


def clean_generated_story(generated_text, prompt):
    story = generated_text.strip()

    if story.startswith(prompt):
        story = story[len(prompt) :].strip()

    if "Story:" in story:
        story = story.rsplit("Story:", 1)[-1].strip()

    story = re.sub(r"<\|[^|]+\|>", "", story)
    story = re.sub(r"\[[^\]]*\]", "", story)
    story = story.replace("Â", "")
    story = re.split(
        r"\b(?:Image description|Requirements|Prompt|User|Assistant|System):",
        story,
    )[0]
    story = re.sub(r"\s+", " ", story).strip()

    return trim_to_word_limit(story)


def extract_generated_text(result):
    generated_text = result["generated_text"]

    if isinstance(generated_text, list):
        return generated_text[-1]["content"]

    return generated_text


def caption_keywords(caption):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", caption.lower())
    return [word for word in words if word not in GENERIC_CAPTION_WORDS]


def keeps_image_context(story, caption):
    keywords = caption_keywords(caption)
    if not keywords:
        return True

    lower_story = story.lower()
    return any(keyword in lower_story for keyword in keywords)


def is_safe_story(story, caption):
    lower_story = story.lower()
    word_total = count_words(story)

    if word_total < MIN_WORDS or word_total > MAX_WORDS:
        return False

    if any(word in lower_story for word in UNSAFE_STORY_WORDS):
        return False

    if re.search(r"\([A-Z][^)]+\)", story):
        return False

    if not keeps_image_context(story, caption):
        return False

    if lower_story.startswith(("write ", "requirements", "image description")):
        return False

    meta_phrases = (
        "the story should",
        "should be about",
        "no other details",
        "return only",
        "based only on",
        "must be",
    )
    if any(phrase in lower_story for phrase in meta_phrases):
        return False

    return story.endswith((".", "!", "?"))


def text2story(caption):
    story_pipe = load_story_model()
    prompt = build_story_prompt(caption)
    messages = [
        {
            "role": "system",
            "content": (
                "You write short, safe, cheerful stories for young children. "
                "You never explain the instructions. You only write the story."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    story_results = story_pipe(
        messages,
        max_new_tokens=130,
        num_return_sequences=4,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        pad_token_id=story_pipe.tokenizer.eos_token_id,
        return_full_text=False,
    )

    for result in story_results:
        story = clean_generated_story(extract_generated_text(result), prompt)
        if is_safe_story(story, caption):
            return story

    return None


def text2audio(story_text):
    audio_pipe = load_text2audio_model()
    return audio_pipe(story_text)


def main():
    st.set_page_config(
        page_title="Image to Audio Story",
        page_icon=":book:",
    )

    st.title("Image to Audio Storytelling App")

    uploaded_file = st.file_uploader(
        "Select an image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Reading the image..."):
            scenario = img2text(image)

        st.write(f"**Image Description:** {scenario}")

        with st.spinner("Generating a child-friendly story..."):
            story = text2story(scenario)

        if not story:
            st.error(
                "The model could not create a safe story this time. "
                "Please try again."
            )
            st.stop()

        st.write(f"**Generated Story:** {story}")
        st.caption(f"Word count: {count_words(story)}")

        with st.spinner("Generating audio..."):
            audio_data = text2audio(story)

        st.audio(
            audio_data["audio"],
            sample_rate=audio_data["sampling_rate"],
        )


if __name__ == "__main__":
    main()
