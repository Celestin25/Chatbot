import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Define dataset directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load necessary models and data
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

# Load datasets
training_dataset = pd.read_csv(f'{working_dir}/Training.csv')
test_dataset = pd.read_csv(f'{working_dir}/Testing.csv')
doc_dataset = pd.read_csv(f'{working_dir}/doctors_dataset.csv', names=['Name', 'Description'])

# Preprocessing
X = training_dataset.iloc[:, 0:132].values
Y = training_dataset.iloc[:, -1].values
dimensionality_reduction = training_dataset.groupby(training_dataset['prognosis']).max()
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(Y)

# Train the classifier
classifier = DecisionTreeClassifier()
classifier.fit(X, y)

cols = training_dataset.columns[:-1]  # Assuming all columns except the last one are features

# Dataset selection for BERT
intents_file_path = os.path.join(working_dir, 'intents.json')
try:
    with open(intents_file_path, 'r') as file:
        intents_data = json.load(file)
except FileNotFoundError:
    st.error("The 'intents.json' file was not found. Please ensure it is placed in the root directory.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding the 'intents.json' file. Please ensure it is in the correct JSON format.")
    st.stop()

# Function to get chatbot response
def get_chatbot_response(user_query):
    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            if pattern.lower() in user_query.lower():
                return intent['responses'][0]  # Return the first response
    return "I'm sorry, I don't have an answer to that question. Please consult a professional."

# Convert mental health data to dataframe for BERT processing
mental_health_df = pd.DataFrame(mental_health_data.items(), columns=['Question', 'Answer'])

# Text cleaning and tokenization for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(data, tokenizer, max_length=256):
    inputs = tokenizer(
        text=data['Question'].tolist(),
        text_pair=data['Answer'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return inputs

inputs = preprocess_data(mental_health_df, tokenizer)
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

# Fine-tune the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Create dataloaders
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Training setup
optimizer = AdamW(model.parameters(), lr=5e-5)


model.train()
for epoch in range(3):  
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        input_ids, attention_mask, token_type_ids = batch
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation loop
model.eval()
all_preds, all_labels = [], []
for batch in val_dataloader:
    input_ids, attention_mask, token_type_ids = batch
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    all_preds.extend(preds)
    all_labels.extend(labels.cpu().numpy())

# Performance metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

st.write(f"Accuracy: {accuracy}")
st.write(f"Precision: {precision}")
st.write(f"Recall: {recall}")
st.write(f"F1 Score: {f1}")

# Streamlit setup
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

with st.sidebar:
    selected = option_menu('Disease Prediction System', 
                           ['Heart Disease Prediction', 'Health Chatbot', 'Mental Health Q&A'], 
                           menu_icon='hospital-fill', 
                           icons=['heart', 'chat', 'info-circle'], 
                           default_index=0)

if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    st.markdown("Please fill out the following details to predict the presence of heart disease.")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120)
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=300)
        restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Probable or Definite Left Ventricular Hypertrophy"}[x])
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, step=0.1)
        ca = st.number_input('Major Vessels Colored by Flouroscopy', min_value=0, max_value=4)

    with col2:
        sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: {0: "Female", 1: "Male"}[x])
        chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=100, max_value=700)
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=250)
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])

    with col3:
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}[x])
        fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: {0: "False", 1: "True"}[x])
        exang = st.radio('Exercise Induced Angina', options=[0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])
        thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3], format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Other"}[x])

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        try:
            heart_prediction = heart_disease_model.predict([user_input])
            heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
            st.success(heart_diagnosis)
        except Exception as e:
            st.error(f"Error in prediction: {e}")

elif selected == 'Health Chatbot':
    st.title('Health Chatbot for Disease Diagnosis')
    st.write("Hey, I am HealthChatbot that can help you to know your disease. How may I help you today?")

    if 'current_node' not in st.session_state:
        st.session_state.current_node = 0
        st.session_state.symptoms_present = []

    def print_disease(node):
        node = node[0]
        val = node.nonzero()
        disease = labelencoder.inverse_transform(val[0])
        return disease

    def recurse(node, depth):
        global tree_, feature_name
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            st.write(f"Do you have {name}?")
            ans = st.radio("Your Answer:", ["yes", "no"], key=f"answer_{depth}")
            if st.button('Next', key=f'next_{depth}'):
                if ans == "yes":
                    st.session_state.symptoms_present.append(name)
                    recurse(tree_.children_left[node], depth+1)
                else:
                    recurse(tree_.children_right[node], depth+1)
        else:
            present_disease = print_disease(tree_.value[node])
            st.write("You may have " + present_disease[0])
            red_cols = dimensionality_reduction.columns
            symptoms_given = red_cols[dimensionality_reduction.loc[present_disease[0]].values[0].nonzero()]
            st.write("Symptoms present: " + str(st.session_state.symptoms_present))
            st.write("Symptoms given: " + str(list(symptoms_given)))
            confidence_level = (1.0*len(st.session_state.symptoms_present))/len(list(symptoms_given))
            st.write("Confidence level is " + str(confidence_level))

            doc_index = [i for i in range(len(disease_desc)) if disease_desc['Disease'][i] == present_disease[0]]
            doc_list = disease_desc.iloc[doc_index[0]]['Description']
            st.write(f"Description: {doc_list}")

            doc_names = doc_dataset['Name'].tolist()
            doc_descs = doc_dataset['Description'].tolist()
            st.write("Doctors nearby:")
            for name, desc in zip(doc_names, doc_descs):
                st.write(f"{name}: {desc}")

    tree_ = classifier.tree_
    feature_name = cols
    recurse(0, 1)

elif selected == 'Mental Health Q&A':
    st.title('Mental Health Q&A Chatbot')
    st.write("Welcome to the Mental Health Q&A Chatbot. Ask me anything about mental health!")

    user_query = st.text_input("Ask your question:")
    if user_query:
        inputs = tokenizer([user_query], padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        response = torch.argmax(outputs.logits, dim=-1).item()

        if response == 1:  # Assuming 1 is the label for relevant/positive answers
            st.write("Based on my knowledge, here's what I can tell you: [RELEVANT_ANSWER]")
        else:
            st.write("I'm sorry, I don't have enough information to answer that question.")

# Performance metrics (for display and evaluation)
def display_metrics(y_true, y_pred, stage="Validation"):
    st.write(f"### {stage} Metrics")
    st.write(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    st.write(f"Precision: {precision_score(y_true, y_pred, average='weighted')}")
    st.write(f"Recall: {recall_score(y_true, y_pred, average='weighted')}")
    st.write(f"F1 Score: {f1_score(y_true, y_pred, average='weighted')}")

display_metrics(all_labels, all_preds, stage="Validation")
