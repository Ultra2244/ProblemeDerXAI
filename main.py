import torch
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openxai.experiment_utils import print_summary, load_config, fill_param_dict
from openxai.explainers.perturbation_methods import get_perturb_method
from openxai.dataloader import ReturnLoaders, ReturnTrainTestX
from openxai import LoadModel, Explainer, Evaluator
from openxai import Explainer
from openxai import Evaluator
from openxai.evaluator import ground_truth_metrics, prediction_metrics, stability_metrics

"""
TODO: Dokumentation
    data_name: german/ ...
    ml_model: ann/lr
    method: lime/shap
    metric: PGU/PGI/ ...

    ... restliche Dokumentation. Welche Metriken? Kurze Erklärung?

    Predictive Faithfulness Metrics
    PGI/PGU:
TODO: Clean Code
    Code aufräumen:
        -Redundancies verringern
        -Häufige literale in variablen abspeichern
    Modeldifferenzierung(ann/lr)
    Explainerdifferenzierung(lime/shap)
    Datasetdifferenzierung
"""
def testModelsData(data_name, method, model_name):
    # Get training and test loaders
    try:
        trainloader, testloader = ReturnLoaders(data_name, download=True, batch_size=10)
    except Exception as e:
        print(f"Fehler beim Laden des Datensatzes {data_name}: {e}")
        return

    # Get input instance
    inputs, labels = next(iter(testloader))
    labels = labels.type(torch.int64)

    # Get full train/test FloatTensors and feature metadata
    X_train, X_test, feature_metadata = ReturnTrainTestX(data_name, float_tensor=True, return_feature_metadata=True)

    # Load pretrained ml model
    model = LoadModel(data_name, ml_model=model_name, pretrained=True)
    print_summary(model, trainloader, testloader)
    preds = model(inputs.float()).argmax(1)
    print(f'First 10 predictions: {preds[:10]}')

    # Load config parameters for the explainer
    try:
        param_dict = load_config('experiment_config.json')['explainers'][method]
    except Exception as e:
        print(f"Fehler beim Laden der Konfiguration: {e}")
        param_dict ={}
        
    # IF LIME/IG, the provide X_train
    param_dict = fill_param_dict(method, param_dict, X_train)
    params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
    print(f'{method.upper()} Parameters\n\n' +'\n'.join(params_preview))
    print('Remaining parameters are set to their default values')

    # Compute explanations
    explainer = Explainer(method, model=model, param_dict=param_dict)
    explanations = explainer.get_explanations(inputs.float(), preds).detach().numpy()
    print(explanations[0])


    ############### Calculating metrics ###############
   
    # Print all possible metrics. 
    print('Ground truth metrics: ', ground_truth_metrics)
    print('Prediction metrics: ', prediction_metrics)
    print('Stability metrics: ', stability_metrics)

    ### Load config for prediction metrics
    
    param_dict = load_config('experiment_config.json')['evaluators']['prediction_metrics']
    param_dict['inputs'] = inputs
    param_dict['explanations'] = explanations
    param_dict['feature_metadata'] = feature_metadata
    param_dict['perturb_method'] = get_perturb_method(param_dict['std'], data_name)
    del param_dict['std']

    ### Load config for ground truth metrics
    
    # PRA and RC require a slightly different set of parameters, without keys "k" and "AUC2" and "predictions"
    gparam_dict1 = load_config('experiment_config.json')['evaluators']['ground_truth_metrics']
    gparam_dict1['explanations'] = explanations
    gparam_dict2 = gparam_dict1.copy()
    gparam_dict1['predictions'] = preds
    del gparam_dict2['k'], gparam_dict2['AUC']

    ### Load config for stability metrics
    
    # Initialize explainer for stability metrics
    exp_method = 'grad'
    exp_param_dict = load_config('experiment_config.json')['explainers'][exp_method]
    exp_param_dict = fill_param_dict(exp_method, exp_param_dict, X_train)  # if LIME/IG
    explainer = Explainer(exp_method, model, exp_param_dict)

    # sparam_dict1 contains X_test as inputs for RIS
    # sparam_dict2 contains inputs as inputs for RRS/ROS
    sparam_dict1 = load_config('experiment_config.json')['evaluators']['stability_metrics']
    sparam_dict1['explainer'] = explainer
    sparam_dict1['feature_metadata'] = feature_metadata
    sparam_dict1['perturb_method'] = get_perturb_method(sparam_dict1['std'], data_name)
    del sparam_dict1['std']

    sparam_dict2 = sparam_dict1.copy()
    sparam_dict1['inputs'] = X_test
    sparam_dict2['inputs'] = X_test

    # Print parameters
    # params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
    # print(f'{metric.upper()} Parameters\n\n' +'\n'.join(params_preview))

    for metric in (stability_metrics + prediction_metrics + ground_truth_metrics):
        # Evaluate the metric accross the test inputs/explanations
        # For RIS, we use the whole test data
        # For RRS/ROS, we use the first 10 datapoints, "inputs"
        evaluator = Evaluator(model, metric)
        if metric in stability_metrics:
            if metric == 'RIS':
                score, mean_score = evaluator.evaluate(**sparam_dict1)
            else:
                score, mean_score = evaluator.evaluate(**sparam_dict2)
        elif metric in prediction_metrics:
            score, mean_score = evaluator.evaluate(**param_dict)
        else:
            if metric == 'PRA' or metric == 'RC':      
                score, mean_score = evaluator.evaluate(**gparam_dict2)
            else:
                score, mean_score = evaluator.evaluate(**gparam_dict1)

        # Calculate standard error
        std_err = np.std(score) / np.sqrt(len(score))

        writeToCsv(data_name, method, metric, mean_score, std_err)

def writeToCsv(data_name, method, metric, mean_score, std_err):
    data = {
        "Dataset": [data_name],
        "Method": [method],
        "Metric": [metric],
        "Mean Score": [round(mean_score, 2)],
        "Standard Error": [round(std_err, 2)]
    }
    excel_file = "results/experiment_results.xlsx"
    sheet_name = "experiment_results"
    df = pd.DataFrame(data)
    try:
        # Falls die Datei schon existiert, öffne sie und füge neue Zeile hinzu
        with pd.ExcelWriter(excel_file, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            df.to_excel(writer, sheet_name=sheet_name, startrow=writer.sheets[sheet_name].max_row, index=False, header=False)
    except FileNotFoundError:
        # Falls die Datei nicht existiert, erstelle eine neue Datei mit Kopfzeilen
        with pd.ExcelWriter(excel_file, mode="w", engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == '__main__':
    for data in ["german", "adult"]:
        for method in ["ann", "lr"]:
            for model_name in ["lime", "shap"]:
                testModelsData('adult', 'lime', 'lr')