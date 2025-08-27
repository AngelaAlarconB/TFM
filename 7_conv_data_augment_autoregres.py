import os

# Establecemos una semilla para reproducibilidad
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1' 
os.environ['TF_CUDNN_DETERMINISTIC'] = '1' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import cv2
import warnings
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler
import joblib
import random
from tensorflow.keras.initializers import GlorotUniform

warnings.filterwarnings('ignore')  # Ignoramos advertencias menores para claridad
tf.config.experimental.enable_op_determinism()

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

initializer = GlorotUniform(seed=SEED)

# Configuración
EXCLUDED_TIME_STEPS = 300
WINDOW_SIZE = 1
BATCH_SIZE = 8
EPOCHS = 50
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

# Función para preparar datos con ventana deslizante (sin normalización)
def create_sliding_window_spatial(data, window_size=1, start_t=0, end_t=1200):
    """
    Crea X (ventana de pasos de tiempo anteriores) e y (valor futuro) para series temporales con estructura espacial.
    
    Args:
        data: Arreglo de datos con forma (H, W, n_times).
        window_size: Tamaño de la ventana temporal.
        start_t: Índice de tiempo inicial (ajustado a 1050 para excluir las primeras 300 muestras).
        end_t: Índice de tiempo final.
    
    Returns:
        X: Características (n_frames, window_size, H, W, 1).
        y: Objetivo (n_frames, H, W, 1).
    """
    X_list = []
    y_list = []
    for t in range(start_t, end_t - window_size):
        X_list.append(data[:, :, t:t + window_size])
        y_list.append(data[:, :, t + window_size])  
    X = np.stack(X_list, axis=0) 
    y = np.stack(y_list, axis=0) 

    X = X.transpose(0, 3, 1, 2)[:, :, :, :, np.newaxis] 
    y = y[:, :, :, np.newaxis]
    return X, y

# Augmentation data
def augment_data(X, y, noise_sigma=0.01, dropout_p=0.05):
    X_aug, y_aug = [X[i] for i in range(X.shape[0])], [y[i] for i in range(y.shape[0])]
    
    for i in range(X.shape[0]):
        # Ruido gaussiano adaptativo
        if np.random.rand() > 0.5:
            sigma = noise_sigma * np.std(X[i])
            noise = np.random.normal(0, sigma, size=X[i].shape)
            X_aug.append(X[i] + noise); y_aug.append(y[i])
        
        # Perturbación temporal (shift de ±1 timestep)
        if np.random.rand() > 0.7:
            shift = np.random.choice([-1, 1])
            X_shift = np.roll(X[i], shift, axis=0)
            X_aug.append(X_shift); y_aug.append(y[i])
        
        # Dropout espacial simulado
        if np.random.rand() > 0.8:
            mask = np.random.binomial(1, 1 - dropout_p, size=X[i].shape)
            X_aug.append(X[i] * mask); y_aug.append(y[i])
        
        # Escalado global
        if np.random.rand() > 0.8:
            scale = np.random.uniform(0.95, 1.05)
            X_aug.append(X[i] * scale); y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

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

# Seleccionamos solo la primera capa (layer=0) y excluimos las primeras 300 muestras
print("Procesando capa 1...")
pa_layer = sample_reshaped[:, :, 0, EXCLUDED_TIME_STEPS:]

# Submuestreo espacial
PA_downsampled = zoom(pa_layer, (0.5, 0.5, 1), order=3)

# Crear ventanas deslizantes
X, y = create_sliding_window_spatial(PA_downsampled, WINDOW_SIZE) 

# Dividir datos
split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Estandarizar datos
scaler_X = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

scaler_y = StandardScaler()
y_train_reshaped = y_train.reshape(-1, 1)
y_train_scaled = scaler_y.fit_transform(y_train_reshaped).reshape(y_train.shape)
y_test_reshaped = y_test.reshape(-1, 1)
y_test_scaled = scaler_y.transform(y_test_reshaped).reshape(y_test.shape)

# Data augmentation (después de escalado para ruido consistente)
X_train_scaled, y_train_scaled = augment_data(X_train_scaled, y_train_scaled)

# Definir modelo Conv2D
def create_conv2d_model(H, W):
    model = Sequential([
        Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=initializer,
               input_shape=(H, W, 1), data_format='channels_last'),
        Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', data_format='channels_last', kernel_initializer=initializer),
        Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='linear', data_format='channels_last', kernel_initializer=initializer)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Crear y entrenar modelo
n_frames, H, W = WINDOW_SIZE, 116, 116
model = create_conv2d_model(H, W)
model.summary()

# Entrenamos el modelo
print("Entrenando Conv2D...")
history = model.fit(X_train_scaled.squeeze(axis=1), y_train_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=False, verbose=1)

# Guardar el modelo
script_name = os.path.splitext(os.path.basename(__file__))[0]
model_dir = f"resultados_{script_name}"
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "model.keras"))
joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y.pkl"))

# Predecir autoregresivamente
y_pred = np.zeros_like(y_test_scaled) 
current_input = X_test_scaled[0:1].squeeze(axis=1)  
for t in range(y_test_scaled.shape[0]):
    pred = model.predict(current_input, verbose=0) 
    y_pred[t] = pred[0]
    current_input = pred
print(f"y_pred shape: {y_pred.shape}")
print(f"y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"y_test_scaled range: [{y_test_scaled.min():.4f}, {y_test_scaled.max():.4f}]")

# Calcular métricas normalizadas
mse_global_scaled = mean_squared_error(y_test_scaled.flatten(), y_pred.flatten())
mae_global_scaled = mean_absolute_error(y_test_scaled.flatten(), y_pred.flatten())
r2_global_scaled = r2_score(y_test_scaled.flatten(), y_pred.flatten())

mse_per_t_scaled = [mean_squared_error(y_test_scaled[t].flatten(), y_pred[t].flatten()) for t in range(y_test_scaled.shape[0])]

# SSIM normalizado
data_range_scaled = y_test_scaled.max() - y_test_scaled.min()
ssim_scores_scaled = [ssim(y_test_scaled[t, :, :, 0], y_pred[t, :, :, 0], data_range=data_range_scaled) 
               for t in range(y_test_scaled.shape[0])]
mean_ssim_scaled = np.mean(ssim_scores_scaled)

y_test_denorm = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test_scaled.shape)
y_pred_denorm = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

# Calcular métricas
mse_global = mean_squared_error(y_test_denorm.flatten(), y_pred_denorm.flatten())
mae_global = mean_absolute_error(y_test_denorm.flatten(), y_pred_denorm.flatten())
r2_global = r2_score(y_test_denorm.flatten(), y_pred_denorm.flatten())

mse_per_t = [mean_squared_error(y_test_denorm[t].flatten(), y_pred_denorm[t].flatten()) for t in range(y_test_denorm.shape[0])]

# SSIM
data_range = y_test_denorm.max() - y_test_denorm.min()
ssim_scores = [ssim(y_test_denorm[t, :, :, 0], y_pred_denorm[t, :, :, 0], data_range=data_range) 
               for t in range(y_test_denorm.shape[0])]
mean_ssim = np.mean(ssim_scores)

# MSE visualization
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(mse_per_t)), 
         mse_per_t, label='Conv2D MSE')
plt.xlabel('Time step')
plt.ylabel('MSE')
plt.title('Mean Squared Error (MSE) per Time Step')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "mse.png"))
plt.close()

# Normalized MSE
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE,
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(mse_per_t_scaled)),
         mse_per_t_scaled, label='Conv2D MSE (normalized)')
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
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(ssim_scores)), 
         ssim_scores, label='Conv2D SSIM')
plt.xlabel('Time step')
plt.ylabel('SSIM')
plt.title('SSIM per Time Step on Test Set')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "ssim.png"))
plt.close()

# Normalized SSIM
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE,
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(ssim_scores_scaled)),
         ssim_scores_scaled, label='Conv2D SSIM (normalized)')
plt.xlabel('Time step')
plt.ylabel('SSIM (normalized)')
plt.title('SSIM per Time Step - Normalized Scale')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "ssim_normalized.png"))
plt.close()

# Visualización de predicciones vs reales (zoom primeros 100 t)
pixel_idx_x, pixel_idx_y = 50, 50
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + 100), 
         y_test_denorm[:100, pixel_idx_x, pixel_idx_y, 0], label='True Values', alpha=0.7)
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + 100), 
         y_pred_denorm[:100, pixel_idx_x, pixel_idx_y, 0], label='Conv2D Predictions', alpha=0.7)
plt.xlabel('Time step')
plt.ylabel('Value')
plt.title(f'Predictions vs True Values (zoom first 100 steps) for Pixel ({pixel_idx_x}, {pixel_idx_y})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "predictions_vs_true_zoom100.png"))
plt.close()

# Real vs Predicted visualization (single pixel)
pixel_idx_x, pixel_idx_y = 50, 50
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + y_test_denorm.shape[0]), 
         y_test_denorm[:, pixel_idx_x, pixel_idx_y, 0], label='True Values', alpha=0.7)
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + y_test_denorm.shape[0]), 
         y_pred_denorm[:, pixel_idx_x, pixel_idx_y, 0], label='Conv2D Predictions', alpha=0.7)
plt.xlabel('Time step')
plt.ylabel('Value')
plt.title(f'Predictions vs True Values for Pixel ({pixel_idx_x}, {pixel_idx_y})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "predictions_vs_true.png"))
plt.close()

# Training loss visualization
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Conv2D Training Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "train_val_loss.png"))
plt.close()

# Crear video con imágenes 2D (real, predicha, error)
frame_width, frame_height = 116, 116
fig_width = frame_width * 3  # Three subplots side by side
fig_height = frame_height
output_video = os.path.join(model_dir, "output_video.mp4")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, FRAME_RATE, (fig_width, fig_height))

# Calcular vmin y vmax para colorbars fijos
vmin_test = y_test_denorm.min()
vmax_test = y_test_denorm.max()
vmin_pred = y_pred_denorm.min()
vmax_pred = y_pred_denorm.max()
error_vmax = np.max(np.abs(y_test_denorm - y_pred_denorm))

# Generate frames and write to video
for t in range(y_test_denorm.shape[0]):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagen real
    im1 = ax1.imshow(y_test_denorm[t, :, :, 0], cmap='viridis', vmin=vmin_test, vmax=vmax_test)
    ax1.set_title('Real')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Imagen predicha
    im2 = ax2.imshow(y_pred_denorm[t, :, :, 0], cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
    ax2.set_title('Predicted')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Error 2D (diferencia absoluta)
    error = np.abs(y_test_denorm[t, :, :, 0] - y_pred_denorm[t, :, :, 0])
    im3 = ax3.imshow(error, cmap='hot', vmin=0, vmax=error_vmax)
    ax3.set_title('2D Error')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Ajustar layout
    plt.suptitle(f'Time step = {EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + t}, SSIM = {ssim_scores[t]:.4f}', y=0.95)
    plt.tight_layout()
    
    # Convertir figura a array NumPy
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convertir de RGB a BGR para OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Redimensionar al tamaño esperado
    frame_resized = cv2.resize(frame_bgr, (fig_width, fig_height), interpolation=cv2.INTER_AREA)
    
    # Escribir frame al video
    out.write(frame_resized)
    plt.close(fig)

# Liberar el escritor de video
out.release()
print(f"Video saved at: {output_video}")

# Guardar métricas
with open(os.path.join(model_dir, "metricas.txt"), "w") as f:
    f.write("Global Results for Conv2D (Denormalized):\n")
    f.write(f"  MSE Global: {mse_global:.6f}\n")
    f.write(f"  MAE Global: {mae_global:.6f}\n")
    f.write(f"  R² Global: {r2_global:.6f}\n")
    f.write(f"  Mean SSIM: {mean_ssim:.6f}\n\n")

    f.write("Global Results for Conv2D (Normalized):\n")
    f.write(f"  MSE Global: {mse_global_scaled:.6f}\n")
    f.write(f"  MAE Global: {mae_global_scaled:.6f}\n")
    f.write(f"  R² Global: {r2_global_scaled:.6f}\n")
    f.write(f"  Mean SSIM: {mean_ssim_scaled:.6f}\n")

# Imprimir métricas
print("\nGlobal Results - Denormalized Scale:")
print(f"  MSE Global: {mse_global:.6f}")
print(f"  MAE Global: {mae_global:.6f}")
print(f"  R² Global: {r2_global:.6f}")
print(f"  Mean SSIM: {mean_ssim:.6f}")

print("\nGlobal Results - Normalized Scale:")
print(f"  MSE Global: {mse_global_scaled:.6f}")
print(f"  MAE Global: {mae_global_scaled:.6f}")
print(f"  R² Global: {r2_global_scaled:.6f}")
print(f"  Mean SSIM: {mean_ssim_scaled:.6f}")