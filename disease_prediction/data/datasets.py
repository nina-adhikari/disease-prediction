import pandas as pd
import numpy as np
import ast
import warnings

DATA_DIR = ''

# Lists of diseases in English and French
DISEASES = ['HIV (initial infection)', 'Whooping cough', 'Chagas',
    'Tuberculosis', 'Influenza',
    'SLE', 'Sarcoidosis', 'Anaphylaxis',
    'Allergic sinusitis', 'Localized edema']

DISEASES_FR = ['VIH (Primo-infection)', 'Coqueluche', 'Chagas',
    'Tuberculose', 'Possible influenza ou syndrome virémique typique',
    'Lupus érythémateux disséminé (LED)', 'Sarcoïdose', 'Anaphylaxie',
    'Rhinite allergique', 'Oedème localisé ou généralisé sans atteinte pulmonaire associée']

# List of symptoms with string entries
SYMPTOMS_WITH_STR_ENTRIES = ['trav1', 'lesion_larger_than_1cm', 'lesions_peeling',
    'pain_char', 'lesion_color', 'pain_somewhere',
    'pain_radiate', 'lesion_location', 'swelling_location']

# Dictionary to replace missing values
REPLACE_DICT = {
    'AGE': 'unknown',
    'pain_char': 'NA',
    'pain_somewhere': 'nowhere',
    'pain_radiate': 'nowhere',
    'pain_intensity': '0',
    'pain_precise': '0',
    'pain_sudden': '0',
    'lesion_color': 'NA',
    'lesion_location': 'nowhere',
    'lesions_peeling': 'N',
    'lesion_pain_swollen': '0',
    'lesion_larger_than_1cm': 'N',
    'lesion_pain_intense': '0',
    'swelling_location': 'nowhere',
    'trav1': 'N',
    'itching_severity': '0'
}

# List of columns containing integer values
INTEGER_COLS = ['AGE', 'pain_intensity', 'pain_precise', 'pain_sudden',
                'lesion_pain_swollen', 'lesion_pain_intense', 'itching_severity']

# Dataframes for conditions and evidences
conditions = pd.DataFrame()
evidences = pd.DataFrame()
evidences_en = pd.DataFrame()

def set_dir(directory):
    """
    Set the directory for data files.

    Args:
    directory (str): Path to the directory.
    """
    global DATA_DIR
    DATA_DIR = directory

def load_metadata(directory = DATA_DIR):
    """
    Load metadata from JSON and CSV files into global dataframes.

    Args:
    directory (str): Path to the directory containing data files.
    """
    global DATA_DIR, conditions, evidences, evidences_en
    DATA_DIR = directory
    conditions = pd.read_json(DATA_DIR + 'release_conditions.json').transpose()
    evidences = pd.read_json(DATA_DIR + 'release_evidences.json').transpose().rename(columns={'possible-values': 'possible_values'})
    evidences_en = pd.read_csv(DATA_DIR + 'evidences_en.csv', index_col=0)

    evidences['possible_values_en'] = [list() for n in range(len(evidences))]

    for row in evidences.itertuples():
        vals = row.possible_values
        if len(row.value_meaning) == 0:
            continue
        for value in vals:
            row.possible_values_en.append(row.value_meaning[value]['en'])

    evidences_en['value_meaning'] = [dict(ast.literal_eval(thing)) for thing in evidences_en['value_meaning'].values]

def get_english(symptom, detail):
    """
    Retrieve the English translation for symptom details.

    Args:
    symptom (str): The symptom.
    detail (str): The detail of the symptom.

    Returns:
    str: The English translation of the detail.
    """
    try:
        val = evidences_en.loc[symptom]['value_meaning'][detail]['en']
    except KeyError:
        val = detail
    return val

def pad_list(l):
    """
    Pad a list to have at least two elements.

    Args:
    l (list): The list to be padded.

    Returns:
    list: The padded list.
    """
    if len(l) >= 2:
        return l
    else:
        return l + [1]

def escape_parentheses(strings):
    """
    Escape parentheses in each string of a list.

    Args:
    strings (list): A list of strings.

    Returns:
    list: A list of strings with escaped parentheses.
    """
    escaped_strings = []
    for string in strings:
        escaped_string = string.replace('(', '\(').replace(')', '\)')
    escaped_strings.append(escaped_string)
    return escaped_strings

class DiagDataFrame(pd.DataFrame):
    """
    Custom DataFrame for diagnosis data processing.
    """

    _metadata = ["ddx"]

    def __init__(self, *args, **kwargs):
        temp_ddx = kwargs.pop('ddx', False)
        super().__init__(*args, **kwargs)
        self.ddx = temp_ddx

    def format_and_translate(self):
        """
        Format and translate diagnosis data.
        """
        if self.ddx:
            self.dds_to_dicts()
        self.evidences_to_lists()
        self.evidences_to_dicts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.expand_evidences()
        self.rename_symptoms()
        self.translate_to_english()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.replace_values()
        self.fillna(0, inplace=True)
        self.to_integers()

    def dds_to_dicts(self):
        """
        Convert 'DIFFERENTIAL_DIAGNOSIS' column to dictionaries.
        """
        self['DIFFERENTIAL_DIAGNOSIS'] = [dict(ast.literal_eval(thing)) for thing in self['DIFFERENTIAL_DIAGNOSIS'].values]

    def evidences_to_lists(self):
        """
        Convert 'EVIDENCES' column to lists.
        """
        self['EVIDENCES'] = [ast.literal_eval(thing) for thing in self['EVIDENCES'].values]

    def evidences_to_dicts(self):
        """
        Convert 'EVIDENCES' column to dictionaries.
        """
        self['EVIDENCES'] = [dict([pad_list(symp.split('_@_')) for symp in symps]) for symps in self['EVIDENCES']]

    def expand_evidences(self):
        """
        Expand 'EVIDENCES' column into separate columns.
        """
        temp = pd.DataFrame(self.pop('EVIDENCES').values.tolist())
        for column in temp.columns:
            self[column] = temp[column]

    def rename_symptoms(self):
        """
        Rename symptoms based on translation.
        """
        renames = {}
        for column in self.columns:
            if column in evidences.index:
                renames[column] = evidences_en.loc[evidences_en['name'] == column].index[0]

        self.rename(columns=renames, inplace=True)

    def translate_to_english(self):
        """
        Translate diagnosis data to English.
        """
        if self.ddx:
            self['DIFFERENTIAL_DIAGNOSIS'] = [{conditions.loc[k]['cond-name-eng']: v.pop(k) for k in list(v.keys())} for v in self['DIFFERENTIAL_DIAGNOSIS']]

        self['PATHOLOGY'] = [conditions.loc[k]['cond-name-eng'] for k in self['PATHOLOGY']]

        self['INITIAL_EVIDENCE'] = [evidences_en.loc[evidences_en['name'] == k].index[0] for k in self['INITIAL_EVIDENCE']]

        for column in SYMPTOMS_WITH_STR_ENTRIES:
            self[column] = [get_english(column, thing) for thing in self[column].values]

    def replace_values(self):
        """
        Replace missing values in the dataframe.
        """
        for column in self.columns:
            if column in REPLACE_DICT:
                self.loc[self[column].isnull(), column] = REPLACE_DICT[column]

    def to_integers(self):
        """
        Convert columns to integer data type.
        """
        for column in self.columns:
            if column in SYMPTOMS_WITH_STR_ENTRIES + ['SEX', 'PATHOLOGY', 'INITIAL_EVIDENCE', 'DIFFERENTIAL_DIAGNOSIS']:
                continue
            self[column] = self[column].astype('int64')

    def _constructor(self, *args, **kwargs):
        return DiagDataFrame(*args, **kwargs)

def load_csv(filename, diseases=DISEASES_FR, ddx=False):
    """
    Load diagnosis data from CSV files.

    Args:
    filename (str): Path to the CSV file.
    diseases (list): List of diseases to filter by.
    ddx (bool): Whether to include differential diagnosis.

    Returns:
    DiagDataFrame: Processed diagnosis data.
    """
    if ddx:
        loader = pd.read_csv(filename, iterator=True, chunksize=10000)
        pattern = '|'.join(escape_parentheses(diseases))
        ddf = DiagDataFrame(
            pd.concat(
                [chunk.loc[(chunk['PATHOLOGY'].isin(diseases)) | (chunk['DIFFERENTIAL_DIAGNOSIS'].str.contains(pattern, regex=True))] for chunk in loader]
                ), ddx=ddx
            )
    else:
        loader = pd.read_csv(filename, iterator=True, chunksize=10000,
                            usecols=lambda x: x != "DIFFERENTIAL_DIAGNOSIS")
        ddf = DiagDataFrame(pd.concat([chunk[chunk['PATHOLOGY'].isin(diseases)] for chunk in loader]), ddx=ddx)
    ddf.format_and_translate()
    return ddf

def load_feather(filename):
    """
    Load diagnosis data from Feather format files.

    Args:
    filename (str): Path to the Feather file.

    Returns:
    DiagDataFrame: Processed diagnosis data.
    """
    return DiagDataFrame(pd.read_feather(filename))

def load_datasets(subsets=['train', 'validate', 'test'], ddx=False, directory=DATA_DIR, csv=False, diseases=DISEASES_FR):
    """
    Load datasets from different subsets.

    Args:
    subsets (list): List of subsets to load.
    ddx (bool): Whether to include differential diagnosis.
    directory (str): Path to the directory containing data files.
    csv (bool): Whether to load data from CSV files.
    diseases (list): List of diseases to filter by.

    Returns:
    dict: Dictionary containing loaded datasets.
    """
    load_metadata(directory)
    df = {}
    for ds in subsets:
        if csv:
            df[ds] = load_csv(directory + 'release_' + ds + '_patients.csv', ddx=ddx, diseases=diseases)
        elif ddx:
            df[ds] = load_feather(directory + ds + '_with_ddx.feather')
        else:
            df[ds] = load_feather(directory + ds + '.feather')
    return df

