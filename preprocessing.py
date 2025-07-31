import pickle as pkl
import streamlit as st
from config import FILENAME, FILEVECTOR

filename = FILENAME
filevector = FILEVECTOR


def unfolding():
    with open(filename, "rb") as modelfile:
        loaded_model = pkl.load(modelfile)

    with open(filevector, "rb") as vec_file:
        loaded_vector = pkl.load(vec_file)
    return loaded_model, loaded_vector


def make_list_sentences(text):
    return [text.lower()]


def run_app():
    with st.form("sentiment analysis "):
        st.title("sentiment analysis ")
        text = st.text_input(" Enter you analysis:")
        submit_button = st.form_submit_button("Analyse")
        loaded_model, loaded_vector = unfolding()

        if submit_button:
            if not text.strip():
                st.warning("please enter you input before hit Analyse")
        if text:
            new_vec = loaded_vector.transform(make_list_sentences(text))
            predict = loaded_model.predict(new_vec)

            if predict[0] == 0:
                st.text(f""" "{text.capitalize()}" is negative tweet""")
            else:
                st.text(f""" "{text.capitalize()}" is positive tweet""")


if __name__ == "__main__":
    run_app()
