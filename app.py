import streamlit as st
import pandas as pd
import numpy as np
import ast
from disease_prediction.data import datasets as ds
import variables

DIR = 'disease_prediction/data/'

ds.load_metadata(DIR)

# questions = [ds.evidences_en['question_en'].loc[symptom] for symptom in variables.SYMPTOMS]
# pain_q = [ds.evidences_en['question_en'].loc[symptom] for symptom in variables.pain_aux]
# lesion_q = [ds.evidences_en['question_en'].loc[symptom] for symptom in variables.lesion_aux]
# swelling_q = [ds.evidences_en['question_en'].loc[symptom] for symptom in variables.swelling_aux]

st.title('Disease prediction from symptoms')

AGE = st.number_input("*What is your age? (Round down to the nearest integer)", min_value=0, max_value=120)

SEX = st.selectbox("*What is your sex? (Only two options available right now)", ('F', 'M'))

initial_titles = list(variables.SYMPTOM_PHRASES.keys())
initial_phrases = list(variables.SYMPTOM_PHRASES.values())

INITIAL_EVIDENCE = st.selectbox(
    label="*What is your main reason for consulting us today?",
    options=initial_titles,
    index=None,
    format_func= (lambda x: variables.SYMPTOM_PHRASES[x])
    )



values = []
aux_values = {
    'pain': [],
    'lesions': [],
    'swelling': [],
}


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

with st.expander("Share additional info (optional):"):
    for i in range(len(variables.SYMPTOMS)):
        symp = variables.SYMPTOMS[i]
        values.append(get_values(symp))
        if symp in aux_values.keys() and values[i]:
            with st.container(border=True):
                for j in range(len(variables.aux[symp])):
                    aux_values[symp].append(get_values(variables.aux[symp][j]))


with st.expander("Summary"):
    st.write("Age: ", AGE)
    st.write("Sex: ", SEX)
    st.write("Other Symptoms:")
    for i in range(len(values)):
        if (values[i] != None and values[i] != False):
            st.write('*', variables.SYMPTOMS[i], ': ', values[i])
    for key in aux_values.keys():
        vals = aux_values[key]
        for i in range(len(vals)):
            if (vals[i] != None and vals[i] != False):
                st.write('*', variables.aux[key][i], ': ', vals[i])
        

def on_submit(button):
    return True

if st.button(
    label=':green-background[Get diagnosis]',
    type='primary',
    #on_click=on_submit,
):
    data = {
        'AGE': AGE,
        'SEX': SEX,
        'INITIAL_EVIDENCE': INITIAL_EVIDENCE
    }
    for i in range(len(variables.SYMPTOMS)):
        if values[i]:
            data[variables.SYMPTOMS[i]] = values[i]
    
    st.write(data)
