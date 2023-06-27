# RMSD-Estimation-Algorithm (REA)
Repository with the RMSD Estimation Algorithm (REA) model, training and testing data.

## Set up conda environment
```
conda create -n rea python==3.9.12
conda activate rea
pip install pandas==1.5.0
pip install tensorflow==2.10.0
```

## Predict
```
python predict.py ./dataset/test_pseudoknotted.csv ./weights/ ./rea_rmsd_predictions_pseudoknotted.csv
```
