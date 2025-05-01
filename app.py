import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
import nltk

# api_key = "AIzaSyAHPpqUignpGcTI1ZfmXfcFcxlpKDtDSrQ"
api_key = "AIzaSyCPfIdMffhoR2nxre5pmCFuYmvEI6G7oyY"


## Classification Part

# Define the image preprocessing transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model and update it to match the saved state
def load_model(model_path, num_classes=4, device=None):
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict the class of a single image
def predict_image(model, image, class_names, device):
    # Load and preprocess the image
    # image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    print(f"Predicted Class: {predicted_class}")
    return predicted_class


# Usage example
model_path = "models/efficientnet_oct_model.pth"
class_names = ['CNV','DME', 'Drusen', 'Normal']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(model_path, num_classes=4, device=device)


## RAG part

os.environ["GOOGLE_API_KEY"] = api_key
model_name= "gemini-1.5-flash"
llm = ChatGoogleGenerativeAI(model=model_name,temperature=0.0, google_api_key=api_key, max_tokens=None)
path = 'Docu/'
glob_pattern = "**/*.txt"
loader = DirectoryLoader(path=path, glob=glob_pattern)
pages = loader.load()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(
            documents=pages, 
            embedding=embeddings
        )

template = '''
You are a specialized ophthalmologist with knowledge in retinal diseases detectable through OCT imaging, particularly Drusen, Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), and normal (healthy) retina. Your goal is to provide your patients with accurate, accessible information based on the type of question asked, focusing on symptoms, causes, diagnosis, treatments, and any preventative measures. Always aim to clarify technical terms in a simple way, and tailor responses to meet the user's needs precisely.
Imagine You are talking to your patient now regarding the disease below

Use the information given below to answer:
{disease}


When answering:
- Be concise and accurate—focus on the main points relevant to the question.
- Provide context if necessary—add any background that might help clarify unfamiliar terms
- Answer directly—if the user asks about symptoms, describe only the symptoms; if they ask for treatment, describe only treatment options.
- dont use any unnecessary symbols like **.
- make sure u give a concise answer not more than 50 words

Examples for Response Style:

User Query: What are Drusen, and are they harmful?
Response: Drusen are yellow deposits that form under the retina, common in aging eyes. They may not initially affect vision, but larger Drusen can indicate a risk for age-related macular degeneration (AMD). Regular check-ups are advised to monitor any changes.
User Query: How is Diabetic Macular Edema (DME) treated?
Response: DME is managed with anti-VEGF injections to reduce fluid build-up in the retina, corticosteroids to decrease inflammation, and, in some cases, laser therapy. Treatment choice depends on the severity and individual patient factors.
User Query: What does a normal OCT scan look like?
Response: A normal OCT scan shows a well-defined retinal structure with no fluid or deposits. Layers are smooth and continuous, with no signs of swelling or abnormal growth.
Use these examples as a guide to ensure clarity, relevance, and simplicity in your responses.

Given User Query:{Question}
Your Response: 

'''
prompt_template = PromptTemplate.from_template(template)

def get_response(docu: str, query: str):
        prompt = prompt_template.format(
            disease=docu, 
            Question=query, 
        )
        response = llm.invoke(prompt)
        return response.content




# Styling for chat bubbles in dark mode
st.markdown("""
       <style>
        /* Set the full page background color */
        body {
            background-color: #e6f7ff; /* Light blue background */
            color: #333; /* Default text color */
        }

        /* Main title styling */
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            width: 100%;
            padding: 10px 0;
            background-color: #005a8d; /* Dark blue */
            color: #ffffff; /* White text */
            border-radius: 10px;
        }

        /* Section headers */
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #003d66; /* Darker blue */
        }

        /* User chat bubble */
        .user-bubble {
            background-color: #004466; /* Dark blue */
            color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: right;
            max-width: 60%;
            float: right;
            clear: both;
        }

        /* AI chat bubble */
        .ai-bubble {
            background-color: #0073e6; /* Brighter blue */
            color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            max-width: 60%;
            float: left;
            clear: both;
        }

        /* Image resizing */
        img {
            max-width: 100%;
            height: auto;
        }

    </style>
""", unsafe_allow_html=True)
st.markdown('<h1 class="main-title">OCT Disease Classification and Query Chatbot</h1>', unsafe_allow_html=True)

Disease_Doc = ""

# Image Upload and Classification Section
st.markdown('<div class="section-header">Upload an OCT Scan</div>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("Choose an OCT image", type=["jpg", "png", "jpeg"])
classification_result = None

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded OCT Image", use_container_width=False)  # Set use_container_width to False for custom size
    
    # Run classification
    with st.spinner("Classifying..."):
        # classification_result = classifier_model.classify(image)  # Replace with actual model call
        
        classification_result = predict_image(model, image, class_names, device)
        d = vectorstore.similarity_search(classification_result,k=1)                       # Example to retrive the contents using the Similarity search
        Disease_Doc = d[0].page_content


    st.success(f"Disease Identified: {classification_result}")

# Chat Interface
st.markdown('<div class="section-header">Disease-Related Query Chat</div>', unsafe_allow_html=True)

# Initialize session state variables if not set
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "new_query_submitted" not in st.session_state:
    st.session_state["new_query_submitted"] = False

if classification_result:
    # Display chat history before showing the input box
    for speaker, message in st.session_state["chat_history"]:
        if speaker == "user":
            st.markdown(f"<div class='user-bubble'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-bubble'>{message}</div>", unsafe_allow_html=True)

    # Input box for the user query
    user_input = st.text_input("Ask a question about your diagnosis:")

    if user_input and not st.session_state["new_query_submitted"]:
        # Set the flag to prevent multiple reruns
        st.session_state["new_query_submitted"] = True

        # Save the user query and generate model answer
        st.session_state["chat_history"].append(("user", user_input))
        
        with st.spinner("Fetching answer..."):
            # answer = rag_model.query(user_input, disease=classification_result)  # Replace with actual RAG model call
            answer = get_response(Disease_Doc, user_input)
            # answer = "Sample answer related to Drusen."  # Placeholder answer for testing
        st.session_state["chat_history"].append(("ai", answer))
        
        # Clear the input box and reset flag after rerun
        st.rerun()

    # Reset the flag after the rerun
    if st.session_state["new_query_submitted"]:
        st.session_state["new_query_submitted"] = False
