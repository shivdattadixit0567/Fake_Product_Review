import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluate import load
from datasets import load_dataset

st.title("Fake Product Review Classification")
st.write("Enter The Review and Predict the Review is Fake or Not")


input_sms = st.text_input("Enter the message")

model_id = "bert-base-uncased"

loaded_model = AutoModelForSequenceClassification.from_pretrained(
    model_id
)
loaded_model.load_state_dict(torch.load('Model.pth',map_location=torch.device('cpu')))

tokenizer = AutoTokenizer.from_pretrained(model_id)


if input_sms is not None:
    
    encoded_input = tokenizer(input_sms, return_tensors='pt')
    output = loaded_model(**encoded_input)
    
    logits = output.logits.detach().numpy()
    prediction = np.argmax(abs(logits))
    print(logits)
    if st.button('Predict'):
        st.subheader("Predictions:")
        if prediction:
            st.write(prediction," :- Review is Fake")
        else:
            st.write(prediction," :- Review is Original")
        st.write(logits)

st.sidebar.markdown("---")
st.sidebar.subheader(" HELLO EVERYONE ")
st.sidebar.text("My Name is Shiv Datta Dixit")
st.sidebar.text('From IIIT BHAGALPUR')
st.sidebar.markdown("[Email:-shivdattadixit0567@gmail.com]")
st.sidebar.markdown("[Link to GitHub Repository:-https://github.com/shivdattadixit0567/Fake_Product_Review_Classification]")
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.subheader("About the Model")
st.sidebar.write("This model is Fake Review Dataset and can be used to Predict the Review is Fake or Not.")
# Optional: Add a link to your GitHub repository or any additional information
st.sidebar.markdown("---")

# Optional: Add a footer

# Run the app
if __name__ == "__main__":
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("1. Write the Review in the text box.")
    st.markdown("2. The model will classify the Review into original or Fake Category.")
