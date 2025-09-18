import torch
import logging

import streamlit as st

from rag import rag

logger = logging.getLogger(__name__)

# Prevent path errors for Torch
torch.classes.__path__ = []

def main():

    st.set_page_config(page_title="AI Document Assistant", page_icon=":books:")
    st.title("AI Document Assistant")

    uploaded_data_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_data_file:
        rag(uploaded_data_file)


if __name__ == "__main__":
    main()
