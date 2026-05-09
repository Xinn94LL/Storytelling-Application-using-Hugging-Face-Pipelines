# Image to Audio Storytelling App

## Project Overview

This project is an Image to Audio Storytelling App built with Streamlit and Hugging Face pipelines. The application allows users to upload an image, generate a short child-friendly story based on the image, and convert the story into audio.

The app is designed for children aged 3 to 10. Therefore, the generated story is intended to be simple, positive, and easy to understand.

## App Workflow

The application follows a three-stage pipeline:

1. **Image to Text**  
   The uploaded image is processed by an image-to-text model to generate a short image description.

2. **Text to Story**  
   The image description is passed into a text-generation model to create a short story of around 50 to 100 words. The story is designed to be suitable for children aged 3 to 10.

3. **Text to Audio**  
   The generated story is converted into audio using a text-to-audio model. Users can listen to the final audio story directly in the Streamlit app.

## Models Used

The app uses the following Hugging Face models:

- **Image-to-Text Model:** `Salesforce/blip-image-captioning-base`
- **Text-Generation Model:** `pranavpsv/genre-story-generator-v2`
- **Text-to-Audio Model:** `Matthijs/mms-tts-eng`

## Main Features

- Upload an image in JPG, JPEG, or PNG format
- Generate an image description
- Generate a short story based on the image description
- Display the generated story and word count
- Convert the story into audio
- Play the generated audio inside the Streamlit app

## Files in This Repository

```text
app.py
requirements.txt
runtime.txt
README.md
