# Disease diagnosis using classification and NLP
#### Authors
Rebecca Ceppas de Castro, Fulya Tastan, Philip Barron, Mohammad Rafiqul Islam, Nina Adhikari, Viraj Meruliya

#### Contents
- [Project Description](#project-description)
- [Web app](#web-app)
- [Using the code](#using-the-code)

## Project Description

### Overview
Automatic Symptom Detection (ASD) and Automatic Diagnosis (AD) have seen several advances in recent years. Patients and medical professionals would benefit from tools that can aid in diagnosing diseases based on antecedents and presenting symptoms. The lack of quality healthcare in many parts of the world makes solving this problem a matter of utmost urgency. The aim of this project is to build a tool that can diagnose a disease based on a list of symptoms and contribute to our understanding of automatic diagnosis.

### Dataset
- Source: [DDXPlus Dataset](https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374) <sup>[1]</sup> 
- Very large synthetic dataset with over 1 million samples spanning 49 unique pathologies.
- Contains information about patient symptoms, their antecedents and socio-demographic data, a true diagnosis, and a differential diagnosis of their underlying conditions. 
- We chose to focus on the following ten diseases:
  - **Infectious Diseases**: HIV (initial infection), Whooping cough, Chagas disease, Tuberculosis, Influenza
  - **Autoimmune and Inflammatory Conditions**: SLE (Systemic Lupus Erythematosus), Sarcoidosis
  - **Allergic Reactions and Related Conditions**: Anaphylaxis, Allergic sinusitis, Localized edema

### Stakeholders
Medical professionals, epidemiology experts, health organizations (e.g. WHO and CDC), data scientists and ML researchers, and end users (patients).

### Key Performance Indicators (KPIs)
- *Precision*: % of predicted positives that are true positives,
- *Recall*: % of true positives that are predicted as positive, and
- *F1 score*: harmonic mean of precision and recall; we want our model to correctly identify patients with a certain disease and also be confident in our prediction at the same time.

### Modeling
- We built several multiclass classification models that identify the disease based on symptoms and antecedents. For final model selection, we adopted the following strategy for training and evaluation:
  - **Features**: We used _ number of features including (initial evidence, pain levels, travel details,...)
  - **Training**: The original had its own train, validation, and test split. To randomize the splits for our models, we combined the training and validation datasets, and then performed an 80% (train) - 20% (validation) split. The different models were trained and validated respectively on these datasets.
  - **Test**: The best model with its optimal hyperparameters was evaluated on the unseen test data.

![flow_chart](https://github.com/nina-adhikari/disease_prediction/assets/59798314/9d4f82f5-af0f-4d15-a434-817929ac1f93)
- We experimented with an alternative approach using natural language processing (NLP), where the data was used to generate 1-3-sentence-long paragraphs of text, and a new dataset was prepared with this text and the disease label. A DistilBERT transformer was fine-tuned on this dataset, and the fine-tuned model was evaluated on the test set. The fine-tuned transformer is available [here](https://huggingface.co/ninaa510/distilbert-finetuned-medical-diagnosis) and the dataset it was trained on [here](https://huggingface.co/datasets/ninaa510/diagnosis-text).

### Results and Outcomes
- Random Forest was found to be the best model, with the following scores:
  - F1: 59.58%
  - Precision: 75.83%
  - Recall: 59.04%
  - Accuracy: 60.40%

A similar performance was also achieved by XGBoost. We chose Random Forest since it is simpler and more interpretable, which would be useful to stakeholders. The feature importance function of Random Forest shows that ‘INITIAL_EVIDENCE’ is the significant input in classifying the disease.
- The fine-tuned text classification transformer achieved an accuracy of 58.68% on the test set.
- We made a web app that can be used to interact with the models, available at [disease-pred.streamlit.app](https://disease-pred.streamlit.app/). 

### Future Directions
- Using more datasets to improve the model performance.
- Building classification models for other diseases in the current dataset.
- Enhancing our app with a chatbot that patients can interact with directly.

### References
1. Fansi Tchango, A., Goel, R., Wen, Z., Martel, J., & Ghosn, J. (2022). DDXPlus Dataset (Version 14).  https://doi.org/10.6084/m9.figshare.20043374.v14

## Web app
The functionality of both our models is available to try out at our [Streamlit web app](https://disease-pred.streamlit.app/).

## Using the code
You can directly install the repository with the following command-line command:

    pip install git+https://github.com/nina-adhikari/disease_prediction
### Classification using the random forest model

If you'd like to use our model directly, all you have to do is:

    import joblib
    rf_pipeline = joblib.load(DIRECTORY + 'random_forest.joblib')
    rf_pipeline.predict_proba(INPUT_DATA)


If you'd like to recreate our steps for training the model, follow along:

**Step 1:** First we import the module and load the datasets as our custom `DiagDataFrame` class:

    from disease_prediction.data import datasets as ds
	
	# The datasets we want to load; you can choose fewer if you'd like
	SUBSETS = ['train', 'validate', 'test']
	
    df = ds.load_datasets(
		    subsets=SUBSETS,
		    directory=DIRECTORY
	)
	    
	for subset in SUBSETS:
		df[subset].set_index('index', inplace=True)
    
where `DIRECTORY` is the directory where you have saved the data files

    disease_prediction/data/train.feather
    disease_prediction/data/validate.feather
    disease_prediction/data/test.feather
The `load_datasets` method has several other functionalities (such as loading from csv, loading a specific collection of diseases, etc.) that you can check out in its documentation.

**Step 2:** Then we carry out a bit of data cleaning pertinent to the specific subset of diseases we have selected:

    d = {'Y': 1, 'N': 0}
    
    # drop the columns that have a single value in all three datasets and convert Y/N to 1/0
    for subset in SUBSETS:
	    df[subset].drop(columns=['pain_radiate', 'lesions_peeling'], inplace=True)
	    df[subset]['lesion_larger_than_1cm'] = df[subset]['lesion_larger_than_1cm'].map(d)

and set up our datasets for training:

    CATEGORICAL_FEATURES = [col for col in df['train'].columns if df['train'][col].dtype == 'object']
    CATEGORICAL_FEATURES.remove('PATHOLOGY')
    
    NUMERICAL_FEATURES = [col for col in df['train'].columns if (set(df['train'][col].unique()) != set([0,1])) and (df['train'][col].dtype != 'object')]
    
    X = {}
    y = {}
    
    for subset in SUBSETS:
    	X[subset] = df[subset].drop(columns=['PATHOLOGY'])
    	y[subset] = df[subset].PATHOLOGY.copy()
**Step 3:** We are now ready to define our random forest pipeline:

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
        
    # optimal model hyperparameters
    n_estimators = 500
    max_depth = 20
    min_samples_leaf = 5
    bootstrap = False
    
    rf = RandomForestClassifier(
	    n_estimators=n_estimators,
	    max_depth=max_depth,
	    min_samples_leaf=min_samples_leaf,
	    bootstrap=bootstrap
    )
    
    rf_pipeline = make_pipeline(
	    ColumnTransformer(
		    [
		    ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
		    ('numerical', StandardScaler(), NUMERICAL_FEATURES)
		    ],
		    remainder='passthrough'),
	    rf
    )
train:

    rf_pipeline.fit(X['train'], y['train'])
and validate/test:

    from sklearn.metrics import classification_report
    print(classification_report(y['validate'], rf_pipeline.predict(X['validate']), digits=4))
    print(classification_report(y['test'], rf_pipeline.predict(X['test']), digits=4))

### Classification using the DistilBERT transformer

If you’d like to use our model directly, all you have to do is:

    from transformers import pipeline
    
    pipe = pipeline("text-classification", model="ninaa510/distilbert-finetuned-medical-diagnosis")
    pipe(TEXT_YOU_WANT_TO_CLASSIFY)

If you’d like to recreate our steps for training the model, follow along:

**Step 1:** We begin by installing the packages we're going to need:

    pip install datasets transformers evaluate imblearn

and loading the modules for text classification:

    from disease_prediction.models import text_classification as tc
    from disease_prediction.models import classification_helper as ch

**Step 2:** Now let's load the data. There are two options here, you can choose the formulaic script-generated text which is in these files (after unzipping the one zip file, of course)

    disease_prediction/data/text-train.zip
    disease_prediction/data/text-validation.json
    disease_prediction/data/text-test.json
or the GPT-3.5-generated text which is in these files:

    disease_prediction/data/text-train-gpt.json
    disease_prediction/data/text-validation-gpt.json
    disease_prediction/data/text-test-gpt.json
We are going to pick the second set of files. There are once again two options. We will describe them separately.

#### Option 3a: Loading from file directly

**Step 3a (1):** We tell the module where our files are:

    ch.DATA_ARGS.train_file = DIRECTORY + 'text-train-gpt.json'
    ch.DATA_ARGS.validation_file= DIRECTORY + 'text-validation-gpt.json'
    ch.DATA_ARGS.test_file = DIRECTORY + 'text-test-gpt.json'
where `DIRECTORY` is once again the directory where you've downloaded the data. We then tell the module which operations we wish to carry out:

    ch.TRAINING_ARGS.do_train = True
    ch.TRAINING_ARGS.do_eval = True
    ch.TRAINING_ARGS.do_predict = False
and how many samples there are in each:

    ch.DATA_ARGS.max_train_samples = N_TRAIN_SAMPLES
    ch.DATA_ARGS.max_val_samples = N_VAL_SAMPLES
    ch.DATA_ARGS.max_test_samples = N_TEST_SAMPLES

**Step 3a (2)**: Finally we are ready to load the data. This function loads the data and also does a bunch of other set up (including transforming the dataset into the right format, tokenizing, etc.):

    tc.setup_from_scratch()

#### Option 3b: Loading from a Pandas dataframe

The second approach is to load from a dictionary of Pandas dataframes. This is the approach we are going to take since we need to do a bit of preprocessing first.

**Step 3b (1):** Load the dataframes:

    import pandas as pd
        
    df = {}
    SUBSETS = ['train', 'validation', 'test']
        
    for subset in SUBSETS:
    	df[subset] = pd.read_json(DIRECTORY + 'text-' + subset + '-gpt.json')

We now create new splits:

    from sklearn.model_selection import train_test_split
    
    df_combined = pd.concat([df['train'], df['validation'], df['test']])
    X_train, X_test, y_train, y_test = train_test_split(df_combined['sentence1'], df_combined['label'], test_size=0.1, random_state=42)

resample to account for class imbalances:

    from imblearn.under_sampling import RandomUnderSampler
    
    rus = RandomUnderSampler(random_state=42)
    
    X_resampled, y_resampled = rus.fit_resample(X_train.to_numpy().reshape(-1, 1), y_train.to_numpy())

re-define our dataframes:

    df_resampled = pd.DataFrame({'sentence1': X_resampled.reshape(-1), 'label': y_resampled})
    
    df['train'] = df_resampled
    df['validation'] = pd.DataFrame({'sentence1': X_test, 'label': y_test})
    df['test'] = pd.DataFrame({'sentence1': X_test, 'label': y_test})
and tell the module how many samples we have:

    ch.DATA_ARGS.max_train_samples = len(X_resampled)
    ch.DATA_ARGS.max_val_samples = len(X_test)
    
    # This is not relevant since we are not predicting anything, but we define it anyway to placate the transformer
    ch.DATA_ARGS.max_test_samples = len(X_test)
**Step 3b (2):** Phew.  We can now load the data and set up the transformer:

    tc.setup_from_scratch(df)

**Step 4:** Time to fine-tune! Optionally, set up the number of epochs you want to train for:

    ch.TRAINING_ARGS.num_train_epochs = NUM_EPOCHS_YOU_WANT
train:

    tc.train()

and evaluate:

    tc.evaluate()

**Step 5:** We can save the fine-tuned model with the following command:

    tc.WRAPPER.save_pretrained(DIRECTORY + 'model')

and load it when needed:

    tc.setup_from_finetuned(DIRECTORY + 'model')
