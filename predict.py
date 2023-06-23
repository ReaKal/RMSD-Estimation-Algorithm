from pandas import read_csv
import numpy as np
import sys
from tensorflow.keras.models import load_model
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("weights_dir") # Directory containing the weights/models
parser.add_argument("output_path")
args = parser.parse_args()

molprobity = read_csv(args.input_path)

# Generating a list of model paths by combinings the weights directory and the model filenames.
models_paths = [os.path.join(args.weights_dir, model) for model in os.listdir(args.weights_dir)]

# Specifying the columns that represent the IDs in the molprobity DataFrame.
id_cols=['rna', 'model', 'run']

# Specifying the features to use for prediction.
features=['Clashscore all atoms',
 'Probably wrong sugar puckers (%)',
 'Bad bonds (%)',
 'Bad angles (%)',
 'Chiral handedness swaps (%)',
 'Tetrahedral geometry outliers (%)',
 'length_target_norm']

predictions = []
# Looping through each path in the models_path list 
# (the best model from every cycle during 10-fold cross-validation with 20 iterations per data split)
for path in models_paths:
    model=load_model(path, compile=False)
    pred = model.predict(molprobity[features], verbose=0).flatten()
    predictions.append(pred)

# Averaging the predictions made by the 10 models.
jury_prediction = np.mean(predictions, axis=0)

predictions_df = molprobity[id_cols]
predictions_df["rea_rms_all"] = jury_prediction
predictions_df.to_csv(args.output_path, index=False)