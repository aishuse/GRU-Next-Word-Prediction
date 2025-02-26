import time
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from PIL import Image

model = load_model("model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

st.title("Next-Word Prediction App")


image = Image.open("img.webp")
st.image(image, use_container_width=True)

st.write("Enter some text and let the model generate the next words!")

text = st.text_input("Enter seed text:", "what is the")
num_words = st.slider("Number of words to generate:", 1, 10, 5)


if st.button("Generate"):
    for _ in range(num_words):
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
        pos = np.argmax(model.predict(padded_token_text))

        for word, index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
                break  

        time.sleep(0.5)

    st.subheader("Generated Text:")
    st.write(text)
