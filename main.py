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
def testModelsData(data_name, method, metric, model):
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
    model = LoadModel(data_name, ml_model=model, pretrained=True)
    print_summary(model, trainloader, testloader)
    preds = model(inputs.float()).argmax(1)
    print(f'First 10 predictions: {preds[:10]}')

    # Load config parameters for the explainer
    try:
        param_dict = load_config('experiment_config.json')['explainers'][method]
    except Exception as e:
        print(f"❌ Fehler beim Laden der Konfiguration: {e}")
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

    # Load config for prediction_metrics
    param_dict = load_config('experiment_config.json')['evaluators']['prediction_metrics']
    param_dict['inputs'] = inputs
    param_dict['explanations'] = explanations
    param_dict['feature_metadata'] = feature_metadata
    param_dict['perturb_method'] = get_perturb_method(param_dict['std'], data_name)
    del param_dict['std']

    # Print parameters
    params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
    print(f'{metric.upper()} Parameters\n\n' +'\n'.join(params_preview))

    for pm in prediction_metrics:
        # Metrik setzen
        metric = pm

        # Evaluate the metric accross the test inputs/explanations
        evaluator = Evaluator(model, metric)
        score, mean_score = evaluator.evaluate(**param_dict)

        # Calculate standard erro
        std_err = np.std(score) / np.sqrt(len(score))
        
        # Print results
        with open("experiment_results.csv", "a") as f:
            f.write(f"{data_name},{method},{metric},{mean_score:.2f},{std_err:.2f}\n")

if __name__ == '__main__':
    testModelsData('german', 'lime', 'PGI', 'ann')