import pandas as pd
import numpy as np
import ast
import warnings

DATA_DIR = ''

SYMPTOMS_WITH_STR_ENTRIES = ['trav1', 'lesion_larger_than_1cm', 'lesions_peeling',
                                'pain_char', 'lesion_color', 'pain_somewhere',
                                'pain_radiate', 'lesion_location', 'swelling_location']

conditions = pd.DataFrame()
evidences = pd.DataFrame()
evidences_en = pd.DataFrame()

def set_dir(directory):
    global DATA_DIR
    DATA_DIR = directory

def load_metadata(directory = DATA_DIR):
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
    try:
        val = evidences_en.loc[symptom]['value_meaning'][detail]['en']
    except KeyError:
        val = detail
    return val

def pad_list(l):
    if len(l) >= 2:
        return l
    else:
        return l + [1]

class DiagDataFrame(pd.DataFrame):

    def __init__(self, csv, ddx = False, *args, **kwargs):
        super().__init__(pd.read_csv(csv), *args, **kwargs)
        self.ddx = ddx
        if self.ddx:
            self.dds_to_dicts()
        else:
            self.drop(columns=['DIFFERENTIAL_DIAGNOSIS'], inplace=True)
        self.evidences_to_lists()
        self.evidences_to_dicts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.expand_evidences()
        self.rename_symptoms()
        self.translate_to_english()
    
    def dds_to_dicts(self):
        self['DIFFERENTIAL_DIAGNOSIS'] = [dict(ast.literal_eval(thing)) for thing in self['DIFFERENTIAL_DIAGNOSIS'].values]

    def evidences_to_lists(self):
        self['EVIDENCES'] = [ast.literal_eval(thing) for thing in self['EVIDENCES'].values]

    def evidences_to_dicts(self):
        self['EVIDENCES'] = [dict([pad_list(symp.split('_@_')) for symp in symps]) for symps in self['EVIDENCES']]

    def expand_evidences(self):
        temp = pd.DataFrame(self.pop('EVIDENCES').values.tolist())
        temp.fillna(0, inplace=True)
        for column in temp.columns:
            self[column] = temp[column]

    def rename_symptoms(self):
        renames = {}
        for column in self.columns:
            if column in evidences.index:
                renames[column] = evidences_en.loc[evidences_en['name'] == column].index[0]

        self.rename(columns=renames, inplace=True)

    def translate_to_english(self):
        if self.ddx:
            self['DIFFERENTIAL_DIAGNOSIS'] = [{conditions.loc[k]['cond-name-eng']: v.pop(k) for k in list(v.keys())} for v in self['DIFFERENTIAL_DIAGNOSIS']]

        self['PATHOLOGY'] = [conditions.loc[k]['cond-name-eng'] for k in self['PATHOLOGY']]

        self['INITIAL_EVIDENCE'] = [evidences_en.loc[evidences_en['name'] == k].index[0] for k in self['INITIAL_EVIDENCE']]

        for column in SYMPTOMS_WITH_STR_ENTRIES:
            self[column] = [get_english(column, thing) for thing in self[column].values]


def load_datasets(subsets=['train', 'validate', 'test'], ddx=False, directory=DATA_DIR):
    load_metadata(directory)
    df = {}
    for ds in subsets:
        df[ds] = DiagDataFrame(DATA_DIR + 'release_' + ds + '_patients.csv', ddx=ddx)
    return df
