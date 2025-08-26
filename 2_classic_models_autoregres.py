import os

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler
import joblib
import random
import cv2

warnings.filterwarnings('ignore')

random.seed(SEED)
np.random.seed(SEED)

# Configuración
EXCLUDED_TIME_STEPS = 300
WINDOW_SIZE = 1
DATA_SHAPE = (107648, 1500)
FRAME_RATE = 10

# Función para verificar dimensiones y datos
def check_data(data, expected_shape=DATA_SHAPE):
    """
    Verifica la integridad de los datos: dimensiones, NaN e infinitos.
    
    Args:
        data: Arreglo de datos (numpy array).
        expected_shape: Forma esperada de los datos.
    
    Returns:
        bool: True si los datos son válidos, False si no.
    """
    if data.shape != expected_shape:
        print(f"Error: Dimensiones de data {data.shape} no coinciden con {expected_shape}.")
        return False
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print("Error: Los datos contienen valores NaN o infinitos.")
        return False
    return True

# Función para preparar datos con ventana deslizante
def create_sliding_window(data, window_size=1, start_t=0, end_t=1200):
    """
    Crea X (ventana de pasos de tiempo anteriores) e y (valor futuro) para series temporales.
    
    Args:
        data: Arreglo de datos con forma (n_pixels, n_times).
        window_size: Tamaño de la ventana temporal.
        start_t: Índice de tiempo inicial.
        end_t: Índice de tiempo final.
    
    Returns:
        X: Características (n_samples, n_pixels, window_size).
        y: Objetivo (n_samples, n_pixels).
    """
    X_list = []
    y_list = []
    for t in range(start_t, end_t - window_size):
        X_list.append(data[:, t:t + window_size])
        y_list.append(data[:, t + window_size])
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y

# Cargar datos
try:
    data = scipy.io.loadmat("datos/RotorSquare_Remod100.mat")
    data_pa = data['PA']
except FileNotFoundError:
    print("Error: No se encontró el archivo.")
    exit()

# Verificamos la integridad de los datos
if not check_data(data_pa):
    exit()

# Reorganizamos los datos
try:
    sample_reshaped = data_pa.reshape(232, 232, 2, 1500)
except ValueError as e:
    print(f"Error en reshape: {e}. Verifica las dimensiones de los datos.")
    exit()

# Seleccionamos la primera capa y excluimos las primeras 300 muestras
print("Procesando capa 1...")
pa_layer = sample_reshaped[:, :, 0, EXCLUDED_TIME_STEPS:] 
PA_downsampled = zoom(pa_layer, (0.5, 0.5, 1), order=3)   
pa_flat = PA_downsampled.reshape(-1, 1200)

# Crear ventanas deslizantes
X, y = create_sliding_window(pa_flat, WINDOW_SIZE)

# Dividir datos
split_idx = int(0.8 * X.shape[0])
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

# Reshape para modelos de ML
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
y_train_reshaped = y_train
y_test_reshaped = y_test 

# Estandarizar datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_reshaped)  
X_test_scaled = scaler_X.transform(X_test_reshaped)       
y_train_scaled = scaler_y.fit_transform(y_train_reshaped) 
y_test_scaled = scaler_y.transform(y_test_reshaped)       

# Definir modelos
models = {
    'Linear Regression': (LinearRegression(), {}),
    'KNN': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]})
}

# Diccionarios para métricas
mse_global = {name: None for name in models}
mae_global = {name: None for name in models}
r2_global = {name: None for name in models}
mse_global_scaled = {name: None for name in models}
mae_global_scaled = {name: None for name in models}
r2_global_scaled = {name: None for name in models}
mse_per_t = {name: [] for name in models}
mse_per_t_scaled = {name: [] for name in models}
ssim_scores = {name: [] for name in models}
ssim_scores_scaled = {name: [] for name in models}
best_models = {}

# Validación cruzada
tscv = TimeSeriesSplit(n_splits=5)

# Entrenar y evaluar
for name, (model, param_grid) in models.items():
    print(f"Entrenando {name}...")
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1)
        grid_search.fit(X_train_scaled, y_train_scaled)
        best_models[name] = grid_search.best_estimator_
        print(f"Mejores parámetros para {name}: {grid_search.best_params_}")
    else:
        model.fit(X_train_scaled, y_train_scaled)
        best_models[name] = model

    # Predicción autoregresiva en escala normalizada
    y_pred_scaled = np.zeros_like(y_test_scaled)
    current_input = X_test_scaled[0:1]
    for t in range(y_test_scaled.shape[0]):
        pred = best_models[name].predict(current_input)
        y_pred_scaled[t] = pred[0]
        current_input = pred

    # Métricas normalizadas
    mse_global_scaled[name] = mean_squared_error(y_test_scaled, y_pred_scaled)
    mae_global_scaled[name] = mean_absolute_error(y_test_scaled, y_pred_scaled)
    r2_global_scaled[name] = r2_score(y_test_scaled, y_pred_scaled)

    mse_per_t_scaled[name] = [mean_squared_error(y_test_scaled[t], y_pred_scaled[t])
                              for t in range(y_test_scaled.shape[0])]

    # Calcular SSIM en normalizado
    y_pred_scaled_reshaped = y_pred_scaled.reshape(y_test_scaled.shape[0], 116, 116, 1)
    y_test_scaled_reshaped = y_test_scaled.reshape(y_test_scaled.shape[0], 116, 116, 1)
    data_range_scaled = y_test_scaled_reshaped.max() - y_test_scaled_reshaped.min()
    ssim_scores_scaled[name] = [ssim(y_test_scaled_reshaped[t, :, :, 0],
                                     y_pred_scaled_reshaped[t, :, :, 0],
                                     data_range=data_range_scaled)
                                for t in range(y_test_scaled.shape[0])]
    
    # Desnormalizar para métricas y visualización
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test_scaled)

    # Métricas globales en escala original
    mse_global[name] = mean_squared_error(y_test_orig, y_pred)
    mae_global[name] = mean_absolute_error(y_test_orig, y_pred)
    r2_global[name] = r2_score(y_test_orig, y_pred)

    # Métricas por tiempo
    mse_per_t[name] = [mean_squared_error(y_test_orig[t], y_pred[t]) for t in range(y_test_orig.shape[0])]

    # Calcular SSIM en escala original
    y_pred_reshaped = y_pred.reshape(y_test_orig.shape[0], 116, 116, 1)
    y_test_reshaped = y_test_orig.reshape(y_test_orig.shape[0], 116, 116, 1)
    data_range = y_test_reshaped.max() - y_test_reshaped.min()
    ssim_scores[name] = [ssim(y_test_reshaped[t, :, :, 0], y_pred_reshaped[t, :, :, 0], data_range=data_range)
                         for t in range(y_test_reshaped.shape[0])]

    # Guardar las predicciones autoregresivas desnormalizadas
    best_models[name].y_pred_autoregresive = y_pred.copy()


# Guardar modelos y escaladores igual que antes
script_name = os.path.splitext(os.path.basename(__file__))[0]
model_dir = f"resultados_{script_name}"
os.makedirs(model_dir, exist_ok=True)
for name, model in best_models.items():
    joblib.dump(model, os.path.join(model_dir, f"{name.replace(' ', '_')}.pkl"))
joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y.pkl"))

# MSE visualization
plt.figure(figsize=(12, 6))
for name, mse_values in mse_per_t.items():
    plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE,
                   EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(mse_values)),
             mse_values, label=f'{name} MSE')
plt.xlabel('Time step')
plt.ylabel('MSE')
plt.title('Mean Squared Error (MSE) per Time Step on Test Set')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "mse.png"))
plt.close()

# Normalized MSE visualization
plt.figure(figsize=(12, 6))
for name, mse_values in mse_per_t_scaled.items():
    plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE,
                   EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(mse_values)),
             mse_values, label=f'{name} MSE (normalized)')
plt.xlabel('Time step')
plt.ylabel('MSE (normalized)')
plt.title('Mean Squared Error (MSE) per Time Step - Normalized Scale')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "mse_normalized.png"))
plt.close()

# SSIM visualization
plt.figure(figsize=(12, 6))
for name, ssim_values in ssim_scores.items():
    plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE,
                   EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(ssim_values)),
             ssim_values, label=f'{name} SSIM')
plt.xlabel('Time step')
plt.ylabel('SSIM')
plt.title('Structural Similarity Index (SSIM) per Time Step on Test Set')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "ssim.png"))
plt.close()

# Normalized SSIM visualization
plt.figure(figsize=(12, 6))
for name, ssim_values in ssim_scores_scaled.items():
    plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE,
                   EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(ssim_values)),
             ssim_values, label=f'{name} SSIM (normalized)')
plt.xlabel('Time step')
plt.ylabel('SSIM (normalized)')
plt.title('SSIM per Time Step - Normalized Scale')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "ssim_normalized.png"))
plt.close()

# Visualización de predicciones vs reales usando predicciones autoregresivas
pixel_idx = 50 * 116 + 50
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE,
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + y_test_orig.shape[0]),
         y_test_orig[:, pixel_idx], label='True Values', alpha=0.7)
for name in best_models:
    y_pred = best_models[name].y_pred_autoregresive  # Usamos la predicción autoregresiva guardada
    plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE,
                   EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + y_test_orig.shape[0]),
             y_pred[:, pixel_idx], label=f'{name} Predictions', alpha=0.7)
plt.xlabel('Time step')
plt.ylabel('Value')
plt.title(f'Predictions vs True Values for Pixel (50, 50)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "predictions_vs_true.png"))
plt.close()

# Crear videos con imágenes 2D (real, predicha, error) para cada modelo usando predicciones autoregresivas
frame_width, frame_height = 116, 116
fig_width = frame_width * 3 
fig_height = frame_height

for model_name in best_models:
    output_video = os.path.join(model_dir, f"output_video_{model_name.replace(' ', '_')}.mp4")
    print(f"Generando video para {model_name}...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, FRAME_RATE, (fig_width, fig_height))

    # Usar predicciones autoregresivas guardadas
    y_pred = best_models[model_name].y_pred_autoregresive
    y_pred_denorm = scaler_y.inverse_transform(y_pred).reshape(y_test.shape[0], 116, 116, 1)
    y_test_denorm = scaler_y.inverse_transform(y_test_scaled).reshape(y_test.shape[0], 116, 116, 1)

    vmin_test = y_test_denorm.min()
    vmax_test = y_test_denorm.max()
    vmin_pred = y_pred_denorm.min()
    vmax_pred = y_pred_denorm.max()
    error_vmax = np.max(np.abs(y_test_denorm - y_pred_denorm))

    for t in range(y_test_denorm.shape[0]):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = ax1.imshow(y_test_denorm[t, :, :, 0], cmap='viridis', vmin=vmin_test, vmax=vmax_test)
        ax1.set_title('Real')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        im2 = ax2.imshow(y_pred_denorm[t, :, :, 0], cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
        ax2.set_title(f'Predicted')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        error = np.abs(y_test_denorm[t, :, :, 0] - y_pred_denorm[t, :, :, 0])
        im3 = ax3.imshow(error, cmap='hot', vmin=0, vmax=error_vmax)
        ax3.set_title('2D Error')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        plt.suptitle(f'Time step = {EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + t}, SSIM = {ssim_scores[model_name][t]:.4f}', y=0.95)
        plt.tight_layout()
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, (fig_width, fig_height), interpolation=cv2.INTER_AREA)
        out.write(frame_resized)
        plt.close(fig)

    out.release()
    print(f"Video saved at: {output_video}")

# Save metrics
with open(os.path.join(model_dir, "metrics.txt"), "w") as f:
    f.write("Global Results (Denormalized):\n")
    for name in best_models:
        f.write(f"{name}:\n")
        f.write(f"  MSE Global: {mse_global[name]:.6f}\n")
        f.write(f"  MAE Global: {mae_global[name]:.6f}\n")
        f.write(f"  R² Global: {r2_global[name]:.6f}\n")
        f.write(f"  Mean SSIM: {np.mean(ssim_scores[name]):.6f}\n")

    f.write("\nGlobal Results (Normalized):\n")
    for name in best_models:
        f.write(f"{name}:\n")
        f.write(f"  MSE Global: {mse_global_scaled[name]:.6f}\n")
        f.write(f"  MAE Global: {mae_global_scaled[name]:.6f}\n")
        f.write(f"  R² Global: {r2_global_scaled[name]:.6f}\n")
        f.write(f"  Mean SSIM: {np.mean(ssim_scores_scaled[name]):.6f}\n")

# Print metrics
print("\nGlobal Results - Denormalized Scale:")
for name in best_models:
    print(f"{name}:")
    print(f"  MSE Global: {mse_global[name]:.6f}")
    print(f"  MAE Global: {mae_global[name]:.6f}")
    print(f"  R² Global: {r2_global[name]:.6f}")
    print(f"  Mean SSIM: {np.mean(ssim_scores[name]):.6f}")

print("\nGlobal Results - Normalized Scale:")
for name in best_models:
    print(f"{name}:")
    print(f"  MSE Global: {mse_global_scaled[name]:.6f}")
    print(f"  MAE Global: {mae_global_scaled[name]:.6f}")
    print(f"  R² Global: {r2_global_scaled[name]:.6f}")
    print(f"  Mean SSIM: {np.mean(ssim_scores_scaled[name]):.6f}")