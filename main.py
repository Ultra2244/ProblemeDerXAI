import torch
import numpy as np
from openxai.experiment_utils import print_summary, load_config, fill_param_dict
from openxai.explainers.perturbation_methods import get_perturb_method
from openxai.dataloader import ReturnLoaders, ReturnTrainTestX
from openxai import LoadModel, Explainer, Evaluator
from openxai import Explainer
from openxai import Evaluator

# Dataset: German Credit, ml_model = 'ann'
def testModelData(data_name, method, metric):
    # Get training and t est loaders
    trainloader, testloader = ReturnLoaders(data_name, download=True, batch_size = 10)
    # Get input instance
    inputs, labels = next(iter(testloader))
    labels = labels.type(torch.int64)

    # Get full train/test FloatTensors and feature metadata
    X_train, X_test, feature_metadata = ReturnTrainTestX(data_name, float_tensor=True, return_feature_metadata=True)

    # Load pretrained ml model
    model = LoadModel(data_name, ml_model='ann', pretrained=True)
    print_summary(model, trainloader, testloader)
    preds = model(inputs.float()).argmax(1)
    print(f'First 10 predictions: {preds[:10]}')

    # Load config parameters for the explainer
    param_dict = load_config('experiment_config.json')['explainers'][method]
    # IF LIME/IG, the provide X_train
    param_dict = fill_param_dict(method, param_dict, X_train)
    params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
    print(f'{method.upper()} Parameters\n\n' +'\n'.join(params_preview))
    print('Remaining parameters are set to their default values')

    # Compute explanations
    explainer = Explainer(method, model=model, param_dict=param_dict)
    explanations = explainer.get_explanations(inputs.float(), preds).detach().numpy()
    print(explanations[0])

    """ # Explanation method with default hyperparameters
    # Load config parameters for the explainer
    param_dict = {} 
    # IF LIME/IG, the provide X_train
    param_dict = fill_param_dict('lime', param_dict, X_train)
    params_preview = params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
    print(f'{'lime'.upper()} Parameters\n\n' +'\n'.join(params_preview))
    print('Remaining parameters are set to their default values')
    lime = Explainer(method, model=model, param_dict=param_dict)
    lime_explanations= lime.get_explanations(inputs, preds).detach().numpy()
    print(lime_explanations[0]) """

    # Choose one of ['PGU', 'PGI']
    # Load config
    param_dict = load_config('experiment_config.json')['evaluators']['prediction_metrics']
    param_dict['inputs'] = inputs
    param_dict['explanations'] = explanations
    param_dict['feature_metadata'] = feature_metadata
    param_dict['perturb_method'] = get_perturb_method(param_dict['std'], data_name)
    del param_dict['std']

    # Print final parameters
    params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
    print(f'{metric.upper()} Parameters\n\n' +'\n'.join(params_preview))

    # Evaluate the metric accross the test inputs/explanations
    evaluator = Evaluator(model, metric)
    score, mean_score = evaluator.evaluate(**param_dict)

    # Print results
    std_err = np.std(score) / np.sqrt(len(score))
    print(f"{metric}: {mean_score:.2f}\u00B1{std_err:.2f}")

if __name__ == '__main__':
    testModelData('german', 'lime', 'PGI')