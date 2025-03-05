import torch
import numpy as np
from openxai.experiment_utils import print_summary, load_config, fill_param_dict
from openxai.explainers.perturbation_methods import get_perturb_method
from openxai.dataloader import ReturnLoaders, ReturnTrainTestX
from openxai import LoadModel, Explainer, Evaluator
from openxai import Explainer
from openxai import Evaluator
from openxai.evaluator import ground_truth_metrics, prediction_metrics, stability_metrics

"""
---TO DO---

data_name: german/ ...
ml_model: ann/lr
method: lime/shap
metric: PGU/PGI/ ...

... restliche Dokumentation. Welche Metriken? Kurze Erklärung?

Predictive Faithfulness Metrics
PGI/PGU:
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


    """
    NOTE:
    inputs muss eigentlich X_test sein
    """
    sparam_dict = load_config('experiment_config.json')['evaluators']['stability_metrics']
    sparam_dict['inputs'] = inputs
    sparam_dict['explainer'] = explainer
    sparam_dict['feature_metadata'] = feature_metadata
    sparam_dict['perturb_method'] = get_perturb_method(sparam_dict['std'], data_name)
    del sparam_dict['std']

    # Print parameters
    # params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
    # print(f'{metric.upper()} Parameters\n\n' +'\n'.join(params_preview))

    # Save results from Stability Metrics into excel sheet
    #for sm in stability_metrics:
    """
    TODO:
    1. Die anderen stability metrics einführen
    2. Code aufräumen:
        -Redundancies verringern
        -Häufige literale in variablen abspeichern
    3. Modeldifferenzierung(ann/lr)
    4. Explainerdifferenzierung(lime/shap)
    5. Datasetdifferenzierung
    """
    metric = 'RIS'

    # Evaluate the metric accross the test inputs/explanations
    evaluator = Evaluator(model, metric)
    score, mean_score = evaluator.evaluate(**sparam_dict)

    # Calculate standard error
    std_err = np.std(score) / np.sqrt(len(score))
    
    # Print results
    with open("results/experiment_results.csv", "a") as f:
        f.write(f"{data_name},{method},{metric},{mean_score:.2f},{std_err:.2f}\n")

    # Save results from Ground Truth Metrics into excel sheet
    for gtm in ground_truth_metrics:
        metric = gtm

        # Evaluate the metric accross the test inputs/explanations
        evaluator = Evaluator(model, metric)
        if metric == 'PRA' or metric == 'RC':      
            score, mean_score = evaluator.evaluate(**gparam_dict2)
        else:
            score, mean_score = evaluator.evaluate(**gparam_dict1)

        # Calculate standard error
        std_err = np.std(score) / np.sqrt(len(score))
        
        # Print results
        with open("results/experiment_results.csv", "a") as f:
            f.write(f"{data_name},{method},{metric},{mean_score:.2f},{std_err:.2f}\n")

    # Save results from Prediction Metrics into excel sheet
    for pm in prediction_metrics:
        metric = pm

        # Evaluate the metric accross the test inputs/explanations
        evaluator = Evaluator(model, metric)
        score, mean_score = evaluator.evaluate(**param_dict)

        # Calculate standard error
        std_err = np.std(score) / np.sqrt(len(score))
        
        # Print results
        with open("results/experiment_results.csv", "a") as f:
            f.write(f"{data_name},{method},{metric},{mean_score:.2f},{std_err:.2f}\n")

if __name__ == '__main__':
    testModelsData('adult', 'lime', 'lr')