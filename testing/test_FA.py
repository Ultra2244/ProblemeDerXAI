# Utils
import torch
import numpy as np
from openxai.experiment_utils import print_summary, load_config, fill_param_dict
from openxai.explainers.perturbation_methods import get_perturb_method

# ML models
from openxai.model import LoadModel

# Data loaders
from openxai.dataloader import ReturnLoaders, ReturnTrainTestX

# Explanation models
from openxai.explainer import Explainer

# Evaluation methods
from openxai.evaluator import Evaluator

# Choose the model and the data set you wish to generate explanations for
n_test_samples = 10
data_name = 'adult' # must be one of ['adult', 'compas', 'gaussian', 'german', 'gmsc', 'heart', 'heloc', 'pima']
model_name = 'lr'    # must be one of ['lr', 'ann']

# Get training and test loaders
trainloader, testloader = ReturnLoaders(data_name=data_name,
                                           download=True,
                                           batch_size=n_test_samples)
inputs, labels = next(iter(testloader))
labels = labels.type(torch.int64)

# Get full train/test FloatTensors and feature metadata
X_train, X_test, feature_metadata = ReturnTrainTestX(data_name, float_tensor=True, return_feature_metadata=True)

# Load pretrained ml model
model = LoadModel(data_name=data_name,
                  ml_model=model_name,
                  pretrained=True)
print_summary(model, trainloader, testloader)
preds = model(inputs.float()).argmax(1)
print(f'First 10 predictions: {preds[:10]}')

# Choose explainer
method = 'lime'

# Load config parameters for the explainer
param_dict = load_config('experiment_config.json')['explainers'][method]

# # If LIME/IG, then provide X_train
param_dict = fill_param_dict(method, param_dict, X_train)
params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
print(f'{method.upper()} Parameters\n\n' +'\n'.join(params_preview))
print('Remaining parameters are set to their default values')

# Compute explanations
lime = Explainer(method, model, param_dict)
lime_exps = lime.get_explanations(inputs, preds).detach().numpy()
print(lime_exps[0])

# Print evaluation metrics
from openxai.evaluator import ground_truth_metrics, prediction_metrics, stability_metrics
print('Ground truth metrics: ', ground_truth_metrics)
print('Prediction metrics: ', prediction_metrics)
print('Stability metrics: ', stability_metrics)

# Choose one of ['PRA', 'RC', 'FA', 'RA', 'SA', 'SRA']
metric = 'FA'  

# Load config
param_dict = load_config('experiment_config.json')['evaluators']['ground_truth_metrics']
param_dict['explanations'] = lime_exps
if metric in ['FA', 'RA', 'SA', 'SRA']:
    param_dict['predictions'] = preds  # flips ground truth according to prediction
elif metric in ['PRA', 'RC']:
    del param_dict['k'], param_dict['AUC']  # not needed for PRA/RC

# Print final parameters
params_preview = [f'{k}: array of size {v.shape}' if hasattr(v, 'shape') else f'{k}: {v}' for k, v in param_dict.items()]
print(f'{metric.upper()} Parameters\n\n' +'\n'.join(params_preview))

# Evaluate the metric across the test inputs/explanations
evaluator = Evaluator(model, metric)
score, mean_score = evaluator.evaluate(**param_dict)

# Print results
std_err = np.std(score) / np.sqrt(len(score))
print(f"{metric}: {mean_score:.2f}\u00B1{std_err:.2f}")
if metric in stability_metrics:
    log_mu, log_std = np.log(mean_score), np.log(std_err)
    print(f"log({metric}): {log_mu:.2f}\u00B1{log_std:.2f}")