
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from tqdm import tqdm
from GPyOpt.methods import BayesianOptimization
from operator import itemgetter
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import shap
import joblib

import warnings
warnings.filterwarnings("ignore")


# ==================== Load Dataset ====================
data_path = "C:/Users/GKlab/Documents/penelitian zakiah/experiment ml + bo/dataset_Bpy-Phen.csv"
data = pd.read_csv(data_path)
data = data.rename(columns={'[acid]': 'acid', '[metal]': 'metal'})

features = data[['acid', 'metal', 'O/A', 'time', 'T']]

# ==================== Hyperparameter Grid ====================
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 50, 100],
    'learning_rate': [0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.0625, 0.125, 0.25, 0.5, 1]
}

# ==================== Modeling & Evaluation Function ====================
def train_and_evaluate_model(X, y, param_grid, target_name, cv=8, save_plot=True):
    print(f"\nTraining model for: {target_name}")

    model = GridSearchCV(
        XGBRegressor(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0
    )
    model.fit(X, y)

    best_model = model.best_estimator_

    mae_scores = cross_val_score(best_model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    rmse_scores = cross_val_score(best_model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    r2_scores = cross_val_score(best_model, X, y, cv=cv, scoring='r2')
    y_pred = best_model.predict(X)
    r2_full = r2_score(y, y_pred)

    print(f"{target_name} Model Evaluation:")
    print(f"  MAE: {-np.mean(mae_scores):.3f}")
    print(f"  RMSE: {-np.mean(rmse_scores):.3f}")
    print(f"  R² (CV): {np.mean(r2_scores):.3f}")
    print(f"  R² (Full): {r2_full:.3f}")

    if save_plot:
        plt.figure(figsize=(7, 6))
        
        # Scatter actual vs predicted in orange with transparency and white edges
        plt.scatter(y, y_pred, color='blue', alpha=0.5, edgecolors='w', s=60, label='Predicted')
        
        # Identity line (perfect prediction) in purple, dashed, semi-transparent
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, alpha=0.8, label='Ideal')
        
        plt.xlabel(f"Actual {target_name}", fontsize=12)
        plt.ylabel(f"Predicted {target_name}", fontsize=12)
        plt.title(f"Predicted vs Actual for {target_name}", fontsize=14)
        
        # Set black box frame around the plot
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        
        filename = f"{target_name}_Prediction.png"
        plt.savefig(filename, dpi=800, bbox_inches='tight', transparent=True)
        plt.close()
        plt.show()
        print(f"Plot saved as: {filename}")
       
    return best_model


# =================== Train Models for All Targets ====================
models = {}
targets = ['E_Ni', 'E_Co']
cv_map = {'E_Ni': 8, 'E_Co': 8}  

for target in targets:
    y = data[target]
    model = train_and_evaluate_model(features, y, param_grid, target, cv=cv_map.get(target, 8))
    models[target] = model

# ==================== (Optional) Save XGB Models for All Targets ====================
for name, model in models.items():
    joblib.dump(model, f"{name}_XGB_model.pkl")



# ==================== Bayesian Optimization using GPyOpt ====================

# Use trained models from earlier
E_Ni_model = models['E_Ni']
E_Co_model = models['E_Co']

# Define objective function for Bayesian optimization
def objective_function(x):
    """
    Predicts E_Ni and E_Co using trained models,
    then calculates a combined weighted score to minimize.
    """
    # Convert input to DataFrame
    param_df = pd.DataFrame(x, columns=['acid', 'metal', 'O/A', 'time', 'T'])

    # Predict values
    pred_E_Ni = E_Ni_model.predict(param_df)
    pred_E_Co = E_Co_model.predict(param_df)

    # Combine with weights
    weight_E_Ni = 0.90
    weight_E_Co = 0.10
    weighted_score = (weight_E_Co * pred_E_Co) / (weight_E_Ni * pred_E_Ni)

    return weighted_score.reshape(-1, 1)  # Shape: (n_samples, 1)


# Define bounds for input variables
bounds = [
    {
        'name': 'acid',
        'type': 'continuous',
        'domain': (data['acid'].min(), data['acid'].max()),
        'interval': 0.01
    },
    {
        'name': 'metal',
        'type': 'discrete',
        'domain': list(np.unique(data['metal'])),
        'interval': 5  # Optional for discrete; can be used if needed for spacing
    },
    {
        'name': 'O/A',
        'type': 'continuous',
        'domain': (data['O/A'].min(), data['O/A'].max()),
        'interval': 0.01
    },
    {
        'name': 'time',
        'type': 'continuous',
        'domain': (data['time'].min(), data['time'].max()),
        'interval': 1
    },
    {
        'name': 'T',
        'type': 'continuous',
        'domain': (data['T'].min(), data['T'].max()),
        'interval': 5
    }
]


# Define the constraint that the sum of AN_percent, EHA_percent, and BA_percent should be <= 100
constraints = [{'name': 'sum_constraint', 'constraint': 'x[:,2] + x[:,3] + x[:,4] - 100'}]

# Prepare initial data
initial_X = features.values
initial_Y = ((0.10 * data['E_Co']) / (0.90 * data['E_Ni'])).values.reshape(-1, 1)

print("\nStarting Bayesian Optimization...")


# Determine the Best (Lowest) Weighted Score
best_existing_score = initial_Y.min()
print(f"Best existing weighted score (minimized): {best_existing_score:.3f}")

# Number of initial random points and sequential Bayesian optimization iterations
initial_points = 10
total_iterations = 100

# Run Bayesian Optimization
np.random.seed(100)
optimizer = BayesianOptimization(
    f=objective_function,
    domain=bounds,
    constraints=constraints,
    acquisition_type='EI',
    acquisition_jitter=0.01,
    X=initial_X,
    Y=initial_Y
)
optimizer.run_optimization(max_iter=100)


# Run the rest using tqdm
for _ in tqdm(range(total_iterations), desc="Bayesian Optimization Progress"):
    x_next = optimizer.suggest_next_locations()
    y_next = objective_function(x_next)

    # Append new data to existing observations
    optimizer.X = np.vstack((optimizer.X, x_next))
    optimizer.Y = np.vstack((optimizer.Y, y_next))

    # Update the model manually
    optimizer._update_model()

# Print best result
print("Best parameters:", optimizer.X[np.argmin(optimizer.Y)])
print("Best score:", np.min(optimizer.Y))

# Post-process results
tested_params = optimizer.X
tested_scores = optimizer.Y

# Combine and sort by best score
params_and_scores = sorted(zip(tested_params, tested_scores), key=itemgetter(1))

# Check for duplicates and enforce minimum distance
existing_params = features.values


# Diversity enforcement
def is_far_enough(new, existing, min_dist=5):
    return all(euclidean(new, ex) >= min_dist for ex in existing)

filtered_params_and_scores = []
for params, score in params_and_scores:
    if not any(np.allclose(params, row, atol=1e-3) for row in existing_params):  # Check duplicates
        if is_far_enough(params, [p for p, _ in filtered_params_and_scores], min_dist=5):
            filtered_params_and_scores.append((params, score))    

# Display Top 10
print("\nTop 10 suggested parameter sets for new experiments:")
for i, (params, score) in enumerate(filtered_params_and_scores[:10]):
    formatted = [round(p, 2) if not isinstance(p, int) else int(p) for p in params]
    print(f"Set {i+1}: Parameters = {formatted}, Weighted Score = {score[0]:.5f}")

    
# Save results
bo_df = pd.DataFrame([{
    'acid': p[0], 'metal': p[1], 'O/A': p[2], 'time': p[3], 'T': p[4], 'score': s[0]
} for p, s in filtered_params_and_scores[:10]])

bo_df.to_csv("Bayesian_Optimization_GPyOpt_Top10_v1.csv", index=False)
print("\nTop 10 BO results saved to 'Bayesian_Optimization_GPyOpt_Top10_v1.csv")

print("Cycle completed")



# ==================== SHAP ====================
# Load dataset
data_path = "C:/Users/GKlab/Documents/penelitian zakiah/experiment ml + bo/dataset_Bpy-Phen.csv"
data = pd.read_csv(data_path)
data = data.rename(columns={'[acid]': 'acid', '[metal]': 'metal'})
features = data[['acid', 'metal', 'O/A', 'time', 'T']]

# Load trained model for E_Ni
model = joblib.load("E_Ni_XGB_model.pkl")

# Create SHAP TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

# === Bar Plot
plt.clf()
fig = plt.figure()
shap.summary_plot(shap_values, features, plot_type="bar", show=False)
plt.tight_layout()
fig.savefig("E_Ni_SHAP_summary_bar.png", dpi=300)
plt.close(fig)


plt.clf()
fig = plt.figure()
shap.summary_plot(shap_values, features, show=False)
plt.tight_layout()
fig.savefig("E_Ni_SHAP_summary_beeswarm.png", dpi=300)
plt.close(fig)


plt.clf()
fig = plt.figure()
shap.dependence_plot("acid", shap_values, features, show=False)
plt.tight_layout()
fig.savefig("E_Ni_SHAP_dependence_acid.png", dpi=300)
plt.close(fig)


# === Optional: Dependence Plot (example: 'acid')
plt.clf()
fig = plt.figure()
shap.dependence_plot("T", shap_values, features)
plt.tight_layout()
plt.savefig("E_Ni_SHAP_Dependence Plot.png", dpi=300)
plt.close(fig)


# === Optional: Dependence Plot (example: 'acid')
plt.clf()
fig = plt.figure()
shap.dependence_plot("O/A", shap_values, features)
plt.tight_layout()
plt.savefig("E_Ni_SHAP_Dependence Plot.png", dpi=300)
plt.close(fig)


# === Optional: Dependence Plot (example: 'acid')
plt.clf()
fig = plt.figure()
shap.dependence_plot("time", shap_values, features)
plt.tight_layout()
plt.savefig("E_Ni_SHAP_Dependence Plot.png", dpi=300)
plt.close(fig)

