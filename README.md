# Predicting Diseases from Symptoms
Authors: Rebecca Ceppas de Castro, Fulya Tastan, Philip Barron, Mohammad Rafiqul Islam, Nina Adhikari, Viraj Meruliya

### Problem Description
The field of Automatic Symptom Detection (ASD) and Automatic Diagnosis (AD) has seen several advances in recent years. Patients as well as medical professionals would benefit from tools that can aid in diagnosing diseases based on antecedents and presenting symptoms. The lack of quality healthcare in many parts of the world makes solving this problem a matter of utmost urgency. The aim of this project is to build a tool that can diagnose a disease based on a list of symptoms and contribute to our understanding of automatic diagnosis.

### Dataset
Link to the dataset: https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374
Description:
This is a very large synthetic dataset with over 1 million entries. It contains information about patient symptoms, their antecedents and socio-demographic data, a true diagnosis and a differential diagnosis of their underlying conditions. The entire dataset includes 49 unique pathologies, ranging from bone fractures to anemia and sarcoidosis. We want to focus on a specific subset of these diseases so we have chosen immune and infectious diseases. This subset includes a total of 11 diseases, which are:
- Infectious Diseases: HIV (initial infection), Whooping cough, Chagas disease, Tuberculosis, Ebola, Influenza
- Autoimmune and Inflammatory Conditions: SLE (Systemic Lupus Erythematosus), Sarcoidosis
- Allergic Reactions and Related Conditions: Anaphylaxis, Allergic sinusitis, Localized edema

### Stakeholders
- Medical professionals
- Healthcare and epidemiology experts
- Health organizations (such as the WHO and CDC)
- Data science and machine learning researchers
- End users (patients)

### Key Performance Indicators (KPIs)
- R-squared: accuracy / correlation
- F_beta: calculated from precision and recall. Beta>1 indicates greater weight on recall, while beta<1 indicates greater weight on precision.
- Precision: % of predicted positives that are true positives
- Recall: % of true positives that are predicted as positive
- Cross Entropy
