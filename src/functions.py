import os
import re

import matplotlib.pyplot as plt
import joblib as jl

from DataProcesser import DataProcesser


def display_missing_barchart(dataframe, columns, missing_value):
    
    labels = []
    rects = []

    plt.figure(figsize=(15,6))

    for m in columns:
        x = dataframe[m][dataframe[m] == missing_value].count()
        ax = plt.bar(m, x, color='r')
        rects.append(ax.patches)
        labels.append(f'{x} ({x*100/len(dataframe):0.2f}%)')

    for rect, label in zip(rects, labels):

        rect = rect[0]

        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 2, label, ha='center', va='bottom', fontfamily='sans-serif', fontsize='medium')

    plt.xticks(range(len(columns)), [col.split('_')[1].upper() if len(col.split('_')) > 1 else col.upper() for col in columns])
    plt.box(False)
    plt.yticks([]) 
    plt.title("Dados Missing por Coluna", loc='left')
    plt.show()


def display_distribution_boxchart(dataframe, columns):
    dataframe.loc[:, columns].plot(kind='box', subplots=True, layout=(1,len(columns)), figsize=(30,5))
    plt.show()


def process_and_predict(dataset, model, imputer, drop_cols):
    dp_teste = DataProcesser(X=dataset.drop(drop_cols, axis=1), mean_dict=imputer)
    dataset_teste_processado = dp_teste.process_test_data()

    predictions = model.predict(dataset_teste_processado)

    return predictions


def process_and_predict_proba(dataset, model, imputer, drop_cols):
    dp_teste = DataProcesser(X=dataset.drop(drop_cols, axis=1), mean_dict=imputer)
    dataset_teste_processado = dp_teste.process_test_data()

    predictions = model.predict_proba(dataset_teste_processado)

    return predictions


def _check_version(model_path, model_name):
    """
    Verifys the last version of the saveds models, and create a new version.
    This is necessary to avoid override previous models and allow to recovery in case of bugs.
    
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    model_dir = os.listdir(model_path)

    version = 1
    model_version = f'{model_name}_V{version}.h5'.lower()

    while model_version in model_dir:

        version += 1
        model_version = f'{model_name}_V{version}.h5'.lower()

    return model_path + '/' + model_version


def _check_dir(path):
    if not os.path.exists(path):
        return False
    elif len(os.listdir(path)) > 0:
        return True      
    else:
        return False


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
		

def load_last_model(model_path):

    if _check_dir(model_path):
        model_dir = os.listdir(model_path)
        model_version = sorted(model_dir, key=_natural_key)[-1]

        print(model_path + '/' + model_version)

        return jl.load(model_path + '/' + model_version)

    else:
        print('no models in directory to load...')
        

def save_model(model, model_path, model_name):
    final_path = _check_version(model_path, model_name)
    jl.dump(model, final_path)
 
