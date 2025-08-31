import os

# Establecemos una semilla para reproducibilidad
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1' 
os.environ['TF_CUDNN_DETERMINISTIC'] = '1' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
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
WINDOW_SIZE = 20
BATCH_SIZE = 8
EPOCHS = 50
DATA_SHAPE = (107648, 1500)
FRAME_RATE = 10

# Función para verificar dimensiones y datos
def check_data(data, expected_shape=(107648, 1500)):
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

# Función para preparar datos con ventana deslizante manteniendo estructura espacial
def create_sliding_window_spatial(data, window_size, start_t=0, end_t=1200):
    """
    Crea X (ventana de pasos de tiempo anteriores) e y (valor futuro) para series temporales con estructura espacial.
    
    Args:
        data: Arreglo de datos con forma (H, W, n_times, C).
        window_size: Tamaño de la ventana temporal.
        start_t: Índice de tiempo inicial (ajustado a 1050 para excluir las primeras 300 muestras).
        end_t: Índice de tiempo final.
    
    Returns:
        X: Características (n_frames, window_size, H, W, C).
        y: Objetivo (n_frames, H, W, C).
    """
    X_list = []
    y_list = []
    for t in range(start_t, end_t - window_size):
        X_list.append(data[:, :, t:t + window_size, :])
        y_list.append(data[:, :, t + window_size, :])  
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    
    # Reorganizamos para formato ConvLSTM2D: (n_frames, window_size, H, W, C)
    X = X.transpose(0, 3, 1, 2, 4)
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

# Seleccionamos ambas capas y excluimos las primeras 300 muestras, recortando a [:230, :230]
print("Procesando ambas capas...")
pa_layers = sample_reshaped[:230, :230, :, EXCLUDED_TIME_STEPS:]

# Submuestreo espacial
PA_downsampled = zoom(pa_layers, (0.5, 0.5, 1, 1), order=3) 

# Transponemos para (H, W, time, C)
PA_downsampled = PA_downsampled.transpose(0, 1, 3, 2)

# Creamos ventanas deslizantes manteniendo estructura espacial
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
y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
y_train_scaled = scaler_y.fit_transform(y_train_reshaped).reshape(y_train.shape)
y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
y_test_scaled = scaler_y.transform(y_test_reshaped).reshape(y_test.shape)

X_train_scaled, y_train_scaled = augment_data(X_train_scaled, y_train_scaled)

# Definimos el modelo ConvLSTM2D
def create_robust_convlstm(time_steps, H, W, C):
    model = Sequential([
        # Bloque 1
        ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', return_sequences=True,
                   activation='tanh', kernel_initializer='glorot_uniform', 
                   kernel_regularizer=l2(1e-4),
                   input_shape=(time_steps, H, W, C)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Bloque 2
        ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', return_sequences=True,
                   activation='tanh', kernel_initializer='glorot_uniform',
                   kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),

        # Bloque 3
        ConvLSTM2D(filters=16, kernel_size=(3,3), padding='same', return_sequences=False,
                   activation='tanh', kernel_initializer='glorot_uniform',
                   kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),

        # Capa final convolucional → salida con C canales (capas cardíacas)
        Conv2D(filters=C, kernel_size=(3,3), padding='same', activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Creamos el modelo
n_frames, H, W, C = WINDOW_SIZE, PA_downsampled.shape[0], PA_downsampled.shape[1], PA_downsampled.shape[3]
model = create_robust_convlstm(n_frames, H, W, C)
model.summary()

# Entrenamos el modelo
print("Entrenando ConvLSTM2D...")
history = model.fit(X_train_scaled, y_train_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=False, verbose=1)

# Guardar el modelo
script_name = os.path.splitext(os.path.basename(__file__))[0]
model_dir = f"resultados_{script_name}"
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "model.keras"))
joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y.pkl"))

# Predecimos autoregresivamente
y_pred = np.zeros_like(y_test_scaled)
current_input = X_test_scaled[0:1]
for t in range(y_test_scaled.shape[0]):
    pred = model.predict(current_input, verbose=0) 
    y_pred[t] = pred[0]
    current_input = np.concatenate([current_input[:, 1:], pred[:, np.newaxis]], axis=1)
print(f"y_pred shape: {y_pred.shape}")
print(f"y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"y_test_scaled range: [{y_test_scaled.min():.4f}, {y_test_scaled.max():.4f}]")

y_test_denorm = scaler_y.inverse_transform(y_test_scaled.reshape(-1, C)).reshape(y_test_scaled.shape)
y_pred_denorm = scaler_y.inverse_transform(y_pred.reshape(-1, C)).reshape(y_pred.shape)

# Calcular métricas
mse_global = mean_squared_error(y_test_denorm.flatten(), y_pred_denorm.flatten())
mae_global = mean_absolute_error(y_test_denorm.flatten(), y_pred_denorm.flatten())
r2_global = r2_score(y_test_denorm.flatten(), y_pred_denorm.flatten())

mse_per_t = [mean_squared_error(y_test_denorm[t].flatten(), y_pred_denorm[t].flatten()) for t in range(y_test_denorm.shape[0])]

# Calcular SSIM
data_range = y_test_denorm.max() - y_test_denorm.min()
ssim_scores = [ssim(y_test_denorm[t], y_pred_denorm[t], data_range=data_range, channel_axis=-1) 
               for t in range(y_test_denorm.shape[0])]
mean_ssim = np.mean(ssim_scores)
    
# Visualización del MSE por tiempo
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(mse_per_t)), mse_per_t, label='MSE ConvLSTM2D')
plt.xlabel('Tiempo t')
plt.ylabel('MSE (predicción t+1)')
plt.title('Error Cuadrático Medio (MSE) por Tiempo en el Conjunto de Prueba')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "mse.png"))
plt.close()

# Visualización de SSIM
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(ssim_scores)), 
         ssim_scores, label='SSIM ConvLSTM2D')
plt.xlabel('Tiempo t')
plt.ylabel('SSIM (predicción t+1)')
plt.title('SSIM por Tiempo en el Conjunto de Prueba')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "ssim.png"))
plt.close()

# Visualización de predicciones vs reales
pixel_idx_x, pixel_idx_y = 50, 50
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + y_test_denorm.shape[0]), 
         y_test_denorm[:, pixel_idx_x, pixel_idx_y, 0], label='Valores Reales (Capa 1)', alpha=0.7)
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + y_test_denorm.shape[0]), 
         y_pred_denorm[:, pixel_idx_x, pixel_idx_y, 0], label='Predicciones ConvLSTM2D (Capa 1)', alpha=0.7)
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + y_test_denorm.shape[0]), 
         y_test_denorm[:, pixel_idx_x, pixel_idx_y, 1], label='Valores Reales (Capa 2)', alpha=0.7)
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + y_test_denorm.shape[0]), 
         y_pred_denorm[:, pixel_idx_x, pixel_idx_y, 1], label='Predicciones ConvLSTM2D (Capa 2)', alpha=0.7)
plt.xlabel('Tiempo t')
plt.ylabel('Valor')
plt.title(f'Predicciones vs Reales para el Píxel ({pixel_idx_x}, {pixel_idx_y})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "predictions_vs_real.png"))
plt.close()

# Visualización de predicciones vs reales (zoom primeros 100 t)
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + 100), 
         y_test_denorm[:100, pixel_idx_x, pixel_idx_y, 0], label='Valores Reales (Capa 1)', alpha=0.7)
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + 100), 
         y_pred_denorm[:100, pixel_idx_x, pixel_idx_y, 0], label='Predicciones ConvLSTM2D (Capa 1)', alpha=0.7)
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + 100), 
         y_test_denorm[:100, pixel_idx_x, pixel_idx_y, 1], label='Valores Reales (Capa 2)', alpha=0.7)
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + 100), 
         y_pred_denorm[:100, pixel_idx_x, pixel_idx_y, 1], label='Predicciones ConvLSTM2D (Capa 2)', alpha=0.7)
plt.xlabel('Tiempo t')
plt.ylabel('Valor')
plt.title(f'Predicciones vs Reales (Zoom primeros 100 t) para el Píxel ({pixel_idx_x}, {pixel_idx_y})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "predictions_vs_real_zoom100.png"))
plt.close()

# Visualización de la pérdida durante el entrenamiento
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.title('Pérdida durante el Entrenamiento de ConvLSTM2D')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "train_val_loss.png"))
plt.close()

# Crear video con imágenes 2D (real, predicha, error) - usando promedio de capas para visualización
frame_width, frame_height = H, W
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
    real_avg = np.mean(y_test_denorm[t], axis=-1)
    pred_avg = np.mean(y_pred_denorm[t], axis=-1)
    error = np.abs(y_test_denorm[t] - y_pred_denorm[t])
    error_avg = np.mean(error, axis=-1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagen real
    im1 = ax1.imshow(real_avg, cmap='viridis', vmin=vmin_test, vmax=vmax_test)
    ax1.set_title('Real')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Imagen predicha
    im2 = ax2.imshow(pred_avg, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
    ax2.set_title('Predicha')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Error 2D (diferencia absoluta)
    im3 = ax3.imshow(error_avg, cmap='hot', vmin=0, vmax=error_vmax)
    ax3.set_title('Error 2D')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Ajustar layout
    plt.suptitle(f'Tiempo t = {EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + t}, SSIM = {ssim_scores[t]:.4f}', y=0.95)
    plt.tight_layout()
    
    # Convertir figura a array NumPy
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convertir de RGB a BGR para OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Redimensionar al tamaño esperado (H*3, H)
    frame_resized = cv2.resize(frame_bgr, (fig_width, fig_height), interpolation=cv2.INTER_AREA)
    
    # Escribir frame al video
    out.write(frame_resized)
    plt.close(fig)

# Liberar el escritor de video
out.release()
print(f"Video guardado en: {output_video}")

# Guardar métricas
with open(os.path.join(model_dir, "metricas.txt"), "w") as f:
    f.write("Resultados Globales para ConvLSTM2D:\n")
    f.write(f"  MSE Global: {mse_global:.6f}\n")
    f.write(f"  MAE Global: {mae_global:.6f}\n")
    f.write(f"  R² Global: {r2_global:.6f}\n")
    f.write(f"  Mean SSIM: {mean_ssim:.6f}\n")

# Imprimir métricas
print("\nResultados Globales para ConvLSTM2D:")
print(f"  MSE Global: {mse_global:.6f}")
print(f"  MAE Global: {mae_global:.6f}")
print(f"  R² Global: {r2_global:.6f}")
print(f"  Mean SSIM: {mean_ssim:.6f}")