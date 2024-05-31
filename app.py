import streamlit as st
import pandas as pd
import numpy as np
import ast
from disease_prediction.data import datasets as ds
import variables
import joblib
import requests
import time

DIR = 'disease_prediction/data/'


TOKEN = st.secrets['HF_TOKEN']
API_URL = "https://api-inference.huggingface.co/models/ninaa510/distilbert-finetuned-medical-diagnosis"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

ds.load_metadata(DIR)

# questions = [ds.evidences_en['question_en'].loc[symptom] for symptom in variables.SYMPTOMS]
# pain_q = [ds.evidences_en['question_en'].loc[symptom] for symptom in variables.pain_aux]
# lesion_q = [ds.evidences_en['question_en'].loc[symptom] for symptom in variables.lesion_aux]
# swelling_q = [ds.evidences_en['question_en'].loc[symptom] for symptom in variables.swelling_aux]

values = []
aux_values = {
    'pain': [],
    'lesions': [],
    'swelling': [],
}


DISCLAIMER_TEXT = """
            <small>Remember, these results are <b>not</b> medical advice. This model is trained to 
             calculate probabilities for each of a list of ten diseases, all adding up to 1, 
            <i>no matter the input</i>.
            """

rf = joblib.load('disease_prediction/models/random_forest.joblib')

DISEASES = rf.classes_

def predict(inputs):
    if not rf:
        return []
    else:
        dataframe = pd.DataFrame(inputs, index=[0])
        dataframe = dataframe[variables.SYMPTOMS_IN_ORDER]
        values = rf.predict_proba(dataframe.astype(object)).tolist()[0]
        dictionary = dict(zip(values, DISEASES))
        return dictionary

def show_forest_results(data):
    keys = list(data.keys())
    keys.sort(reverse=True)
    for key in keys:
        label = data[key]
        value = key
        if value >= 0.1:
            st.progress(
                text="* " + label + ': ' + str(round(value*100, 2)) + ' \%',
                value=value
                )






st.title('Disease diagnosis from symptoms')

st.write(
    """
    This is a prototype app that uses machine learning to automatically diagnose
    a disease based on a list of symptoms. This app is for educational,
    research and entertainment purposes only. Nothing here is 
    (even remotely) supposed to be medical advice.

    There are two different modes that this app can be used in: the [intake form](#intake-form)
    and the [text form](#text-form-experimental).
    """
    )

st.subheader("Intake form")

st.write(
    """
    This part of the app uses a random forest classifier to predict
    the probability of each of a list of ten diseases based on the
    options you select in the form below. The dataset this classifier
    was trained on can be accessed
    [here](https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374).
    """
)


AGE = st.number_input("*What is your age? (Round down to the nearest integer)", min_value=0, max_value=120)

SEX = st.selectbox("*What is your sex? (Only two options available right now)", ('F', 'M'))

initial_titles = list(variables.SYMPTOM_PHRASES.keys())
initial_phrases = list(variables.SYMPTOM_PHRASES.values())

INITIAL_EVIDENCE = st.selectbox(
    label="*What is your main reason for consulting us today?",
    options=initial_titles,
    index=3,
    format_func= (lambda x: variables.SYMPTOM_PHRASES[x])
)


def get_dt(symptom):
    return ds.evidences_en['data_type'].loc[symptom]

def get_values(symptom):
    dt = get_dt(symptom)
    q = ds.evidences_en['question_en'].loc[symptom]
    if dt == 'B':
        return st.checkbox(q, False)
    else:
        poss_vals_en = ast.literal_eval(ds.evidences_en['possible_values_en'].loc[symptom])
        if len(poss_vals_en) != 0:
            return st.selectbox(q, poss_vals_en, index=None)
        else:
            return st.selectbox(q, ast.literal_eval(ds.evidences_en['possible_values'].loc[symptom]), index=None)

with st.expander("Share additional info:"):
    for i in range(len(variables.SYMPTOMS)):
        symp = variables.SYMPTOMS[i]
        values.append(get_values(symp))
        if symp in aux_values.keys():
            if values[i]:
                with st.container(border=True):
                    for j in range(len(variables.aux[symp])):
                        aux_values[symp].append(get_values(variables.aux[symp][j]))
            else:
                for j in range(len(variables.aux[symp])):
                    aux_values[symp].append(0)


# with st.expander("Summary"):
#     st.write("Age: ", AGE)
#     st.write("Sex: ", SEX)
#     st.write("Other Symptoms:")
#     for i in range(len(values)):
#         if (values[i] != None and values[i] != False):
#             st.write('*', variables.SYMPTOMS[i], ': ', values[i])
#     for key in aux_values.keys():
#         vals = aux_values[key]
#         for i in range(len(vals)):
#             if (vals[i] != None and vals[i] != False):
#                 st.write('*', variables.aux[key][i], ': ', vals[i])



if st.button(
    label='Get diagnosis',
    type='primary',
    #on_click=on_submit,
):
    data = {
        'AGE': AGE,
        'SEX': SEX,
        'INITIAL_EVIDENCE': INITIAL_EVIDENCE
    }
    for i in range(len(variables.SYMPTOMS)):
        data[variables.SYMPTOMS[i]] = values[i]
    for key in aux_values:
        for i in range(len(aux_values[key])):
            data[variables.aux[key][i]] = aux_values[key][i]
    
    for key in data:
        if data[key] == True:
            data[key] = 1
        if data[key] == False or data[key] == None:
            data[key] = 0
    
    answer = predict(data)
    if (list(data.values()).count(0) > 85):
        st.error("Sorry, there isn't enough data to generate a diagnosis. \
                 Please share some more information above.")
    elif answer == []:
        st.error("Sorry, we could not fetch a diagnosis for you right now. \
                 Please try again later.")
    else:
        with st.container(border=True):
            st.write("#### Results")
            st.write("Based on our model, these are your most likely diseases:")
            #st.write('#### ' + answer[0])
            show_forest_results(answer)
            st.divider()
            st.write("#### Disclaimer")
            st.html(DISCLAIMER_TEXT)
            st.divider()
            st.write("We hope you enjoyed using our model. Now please go consult a real doctor.")




# Text entry section


st.subheader("Text form (experimental)")

st.write("""
         This part of the app uses a fine-tuned transformer to classify
         a text input into one of ten diseases. This functionality has a lower accuracy
         than the part above and should be treated with even more caution.
         You can access the fine-tuned transformer [here](https://huggingface.co/ninaa510/distilbert-finetuned-medical-diagnosis)
         and the dataset used to train it
         [here](https://huggingface.co/datasets/ninaa510/diagnosis-text).
         """)

#myform = st.form('myform')

myform = st.container(border=True)

myform.write("Describe your symptoms in plain English using at least 50 characters. \
             You can use the following text as an example:")

myform.write("> *I have had a persistent cough for the last three days. \
             The cough sometimes includes blood. I am also suffering from fatigue \
             and a loss of appetite.*")


txt = myform.text_area("Describe your symptoms.",
                       placeholder='Minimum 50 characters.',
)


def query(payload):
    json = '{ input: ' + payload + ', options={wait_for_model: True} }'
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=1)
    return response.json()

def show_results(data):
    for entry in data[0]:
        continue
    st.write("#### Results")
    st.write("Based on our model, these are your most likely diseases:")
    for entry in data[0]:
        label = entry['label']
        value = entry['score']
        if value >= 0.1:
            st.progress(
                text="* " + label + ': ' + str(round(value*100, 2)) + ' \%',
                value=value
                )
    st.divider()
    st.write("#### Disclaimer")
    st.html(DISCLAIMER_TEXT)
    st.html("<small>For example, the input")
    st.markdown(
        "> <small>*I’m sorry, but I’m a large language model trained by OpenAl, and I don’t have \
            access to the internet or any external information sources. \
            I can only generate responses based on the text that I was trained on, \
            which has a knowledge cutoff of 2021. I can’t provide links to recent news \
            articles or other information that may have been published since then*",
            unsafe_allow_html=True
        )
    st.html("<small>gives the result")
    dis = ['Anaphylaxis', 'HIV (initial infection)', 'SLE', 'Chagas',]
    probs = [28.64, 20.84, 15.6, 12.7]
    for i in range(len(dis)):
        st.progress(
            text=dis[i] + ': ' + str(probs[i]) + ' \%',
            value=probs[i]/100
        )
    st.html("<small>which is obviously nonsense.")

def on_submit():
    if len(str(txt)) < 50:
        st.write("There is too little text to generate output.")
        return
    #st.write(txt)
    data = query(txt)
    try:
        show_results(data)
        return
    except:
        with st.spinner('Waking up the server...'):
            time.sleep(5)
        with st.spinner('Calculating...'):
            time.sleep(5)
    data = query(txt)
    try:
        show_results(data)
        return
    except:
        st.write("Sorry, we could not generate a diagnosis for you. Please try again later.")

if myform.button("Submit", type='primary', ):
    new_text = txt
    with myform.container(border=True):
        on_submit()
        myform.write("We hope you enjoyed using our model. Now please go consult a real doctor.")
    #form_button = False