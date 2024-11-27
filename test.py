import torch
import numpy as np
from openxai.experiment_utils import print_summary, load_config, fill_param_dict
from openxai.explainers.perturbation_methods import get_perturb_method

from openxai.dataloader import ReturnLoaders, ReturnTrainTestX
# Get training and t est loaders
trainloader, testloader = ReturnLoaders(data_name='german', download=True, batch_size = 10)
# Get input instance
inputs, labels = next(iter(testloader))
labels = labels.type(torch.int64)

# Get full train/test FloatTensors and feature metadata
X_train, X_test, feature_metadata = ReturnTrainTestX('german', float_tensor=True, return_feature_metadata=True)

from openxai import LoadModel
# Load pretrained ml model
model = LoadModel(data_name= 'german', ml_model='ann', pretrained=True)
print_summary(model, trainloader, testloader)
preds = model(inputs.float()).argmax(1)
print(f'First 10 predictions: {preds[:10]}')

from openxai import Explainer
# Load config parameters for the explainer
param_dict = load_config('experiment_config.json')['explainers']['lime']
# IF LIME/IG, the provide X_train
param_dict = fill_param_dict('lime', param_dict, X_train)
params_preview = params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
print(f'{'lime'.upper()} Parameters\n\n' +'\n'.join(params_preview))
print('Remaining parameters are set to their default values')
lime = Explainer(method='lime', model=model, param_dict=param_dict)
lime_explanations= lime.get_explanations(inputs, preds).detach().numpy()
print(lime_explanations[0])

# # Explanation method with default hyperparameters
# # Load config parameters for the explainer
# param_dict = {} 
# # IF LIME/IG, the provide X_train
# param_dict = fill_param_dict('lime', param_dict, X_train)
# params_preview = params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
# print(f'{'lime'.upper()} Parameters\n\n' +'\n'.join(params_preview))
# print('Remaining parameters are set to their default values')
# lime = Explainer(method='lime', model=model, param_dict=param_dict)
# lime_explanations= lime.get_explanations(inputs, preds).detach().numpy()
# print(lime_explanations[0])

# Choose one of ['PGU', 'PGI']
# Load config
param_dict = load_config('experiment_config.json')['evaluators']['prediction_metrics']
param_dict['inputs'] = X_test
param_dict['explanations'] = lime_explanations
param_dict['feature_metadata'] = feature_metadata
param_dict['perturb_method'] = get_perturb_method(param_dict['std'], "german")
del param_dict['std']

# Print final parameters
params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
print(f'{'PGI'.upper()} Parameters\n\n' +'\n'.join(params_preview))

from openxai import Evaluator
# Evaluate the metric accross the test inputs/explanations
evaluator = Evaluator(model, metric='PGI')
score, mean_score = evaluator.evaluate(**param_dict)

# Print results
std_err = np.std(score) / np.sqrt(len(score))
print(f"{"PGI"}: {mean_score:.2f}\u00B1{std_err:.2f}")
if "PGI" in ['PGI']:
    log_mu, log_std = np.log(mean_score), np.log(std_err)
    print(f"log({'PGI'}): {log_mu:.2f}\u00B1{log_std:.2f}")
