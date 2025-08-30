import os

# Semilla para reproducibilidad
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, Reshape, Dense, LayerNormalization, Dropout,
                                     Add, Embedding)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import MultiHeadAttention
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom
import cv2
import warnings
import random
import joblib

warnings.filterwarnings('ignore')
tf.config.experimental.enable_op_determinism()

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

initializer = GlorotUniform(seed=SEED)

# Configuración
EXCLUDED_TIME_STEPS = 300
WINDOW_SIZE = 20
BATCH_SIZE = 8
EPOCHS = 100
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

    # Reorganizamos para formato: (n_frames, window_size, H, W, 1)
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

# Modelo Spatiotemporal Transformer para entrenamiento final (salida 4D)
def create_spatiotemporal_transformer_final(n_frames=WINDOW_SIZE, H=116, W=116,
                                            num_heads=4, key_dim=128, num_blocks=4,
                                            dropout=0.1, learning_rate=1e-3, l2_reg=1e-4):
    """
    Entrada: (B, n_frames, H, W, 1)
    1) Aplanamos espacialmente por frame: (B, n_frames, H*W)
    2) Proyectamos a d_model=key_dim
    3) Sumamos embedding temporal
    4) Bloques Transformer (self-attn temporal + MLP)
    5) Tomamos el último token temporal y proyectamos a H*W
    6) Reshape -> (H, W, 1)
    """
    inp = Input(shape=(n_frames, H, W, 1), name="input")
    # (B, n_frames, H*W)
    x = Reshape((n_frames, H * W), name="flatten_spatial")(inp)

    # Proyección a d_model
    x = Dense(key_dim, activation='relu', kernel_regularizer=l2(l2_reg), name="proj")(x)
    x = LayerNormalization(name="ln_in")(x)

    # Positional embedding temporal (aprendible)
    pos_ids = tf.keras.layers.Lambda(lambda t: tf.tile(
        tf.expand_dims(tf.range(n_frames), axis=0), [tf.shape(t)[0], 1]
    ), name="make_pos_ids")(x)  # (B, n_frames)
    pos_emb = Embedding(input_dim=n_frames, output_dim=key_dim, name="temb")(pos_ids)
    x = Add(name="add_pos")([x, pos_emb])

    # Bloques Transformer
    for b in range(num_blocks):
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                      name=f"mha_{b}")(x, x)
        attn_out = Dropout(dropout, name=f"mha_do_{b}")(attn_out)
        x = Add(name=f"res1_{b}")([x, attn_out])
        x = LayerNormalization(name=f"ln1_{b}")(x)

        # Feed-forward (corregido: la segunda Dense toma ff, no x)
        ff = Dense(key_dim * 2, activation='relu', kernel_regularizer=l2(l2_reg),
                   name=f"ff1_{b}")(x)
        ff = Dropout(dropout, name=f"ff_do_{b}")(ff)
        ff = Dense(key_dim, activation='relu', kernel_regularizer=l2(l2_reg),
                   name=f"ff2_{b}")(ff)
        x = Add(name=f"res2_{b}")([x, ff])
        x = LayerNormalization(name=f"ln2_{b}")(x)

    # Usamos el último paso temporal
    last = x[:, -1, :]                                      # (B, key_dim)
    out_vec = Dense(H * W, activation='linear', name="to_pixels")(last)
    out = Reshape((H, W, 1), name="out_2d")(out_vec)

    model = Model(inp, out, name="SpatioTemporalTransformer")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

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

# Reorganizar datos
try:
    sample_reshaped = data_pa.reshape(232, 232, 2, 1500)
except ValueError as e:
    print(f"Error en reshape: {e}.")
    exit()

# Seleccionamos solo la primera capa (layer=0) y excluimos las primeras 300 muestras
print("Procesando capa 1...")
pa_layer = sample_reshaped[:, :, 0, EXCLUDED_TIME_STEPS:]

# Submuestreo espacial
PA_downsampled = zoom(pa_layer, (0.5, 0.5, 1), order=1)

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

# Crear y entrenar modelo
time_steps, H, W = WINDOW_SIZE, 116, 116
model = create_spatiotemporal_transformer_final(
    n_frames=time_steps, H=H, W=W,
    num_heads=4, key_dim=128, num_blocks=4,
    dropout=0.1, learning_rate=1e-3, l2_reg=1e-4
)
model.summary()

print("Entrenando Transformer espaciotemporal...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    shuffle=False,
    verbose=1,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss', min_lr=1e-6)
    ]
)

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

# Calcular métricas normalizadas
mse_global_scaled = mean_squared_error(y_test_scaled.flatten(), y_pred.flatten())
mae_global_scaled = mean_absolute_error(y_test_scaled.flatten(), y_pred.flatten())
r2_global_scaled = r2_score(y_test_scaled.flatten(), y_pred.flatten())

mse_per_t_scaled = [
    mean_squared_error(y_test_scaled[t].flatten(), y_pred[t].flatten())
    for t in range(y_test_scaled.shape[0])
]

# SSIM normalizado
data_range_scaled = y_test_scaled.max() - y_test_scaled.min()
ssim_scores_scaled = [
    ssim(y_test_scaled[t, :, :, 0], y_pred[t, :, :, 0], data_range=data_range_scaled)
    for t in range(y_test_scaled.shape[0])
]
mean_ssim_scaled = np.mean(ssim_scores_scaled)

y_test_denorm = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test_scaled.shape)
y_pred_denorm = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

# Calcular métricas
mse_global = mean_squared_error(y_test_denorm.flatten(), y_pred_denorm.flatten())
mae_global = mean_absolute_error(y_test_denorm.flatten(), y_pred_denorm.flatten())
r2_global = r2_score(y_test_denorm.flatten(), y_pred_denorm.flatten())

mse_per_t = [mean_squared_error(y_test_denorm[t].flatten(), y_pred_denorm[t].flatten()) for t in range(y_test_denorm.shape[0])]

# Calcular SSIM
data_range = y_test_denorm.max() - y_test_denorm.min()
ssim_scores = [ssim(y_test_denorm[t, :, :, 0], y_pred_denorm[t, :, :, 0], data_range=data_range) 
               for t in range(y_test_denorm.shape[0])]
mean_ssim = np.mean(ssim_scores)
    
# MSE visualization
plt.figure(figsize=(12, 6))
plt.plot(range(EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE, 
               EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + len(mse_per_t)), 
         mse_per_t, label='Transformer MSE')
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
         mse_per_t_scaled, label='Transformer MSE (normalized)')
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
         ssim_scores, label='Transformer SSIM')
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
         ssim_scores_scaled, label='Transformer SSIM (normalized)')
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
         y_pred_denorm[:100, pixel_idx_x, pixel_idx_y, 0], label='Transformer Predictions', alpha=0.7)
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
         y_pred_denorm[:, pixel_idx_x, pixel_idx_y, 0], label='Transformer Predictions', alpha=0.7)
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
plt.title('Transformer Training Loss')
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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.48, 1.16), dpi=300)
    
    fontsize_title = 6  
    fontsize_ticks = 4 
    
    # Imagen real
    im1 = ax1.imshow(y_test_denorm[t, :, :, 0], cmap='viridis', vmin=vmin_test, vmax=vmax_test)
    ax1.set_title('Real', fontsize=fontsize_title)
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=fontsize_ticks)
    
    # Imagen predicha
    im2 = ax2.imshow(y_pred_denorm[t, :, :, 0], cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
    ax2.set_title('Predicted', fontsize=fontsize_title)
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=fontsize_ticks)
    
    # Error 2D
    error = np.abs(y_test_denorm[t, :, :, 0] - y_pred_denorm[t, :, :, 0])
    im3 = ax3.imshow(error, cmap='hot', vmin=0, vmax=error_vmax)
    ax3.set_title('2D Error', fontsize=fontsize_title)
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=fontsize_ticks)

    # Ajustar layout
    plt.suptitle(f'Time step = {EXCLUDED_TIME_STEPS + split_idx + WINDOW_SIZE + t}, SSIM = {ssim_scores[t]:.4f}', 
                 y=0.98, fontsize=fontsize_title)
    plt.tight_layout(pad=0.5)  
    
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    frame_resized = cv2.resize(frame_bgr, (fig_width, fig_height), interpolation=cv2.INTER_CUBIC)
    
    out.write(frame_resized)
    plt.close(fig)

# Liberar el escritor de video
out.release()
print(f"Video saved at: {output_video}")

# Guardar métricas
with open(os.path.join(model_dir, "metricas.txt"), "w") as f:
    f.write("Global Results for Transformer (Denormalized):\n")
    f.write(f"  MSE Global: {mse_global:.6f}\n")
    f.write(f"  MAE Global: {mae_global:.6f}\n")
    f.write(f"  R² Global: {r2_global:.6f}\n")
    f.write(f"  Mean SSIM: {mean_ssim:.6f}\n\n")

    f.write("Global Results for Transformer (Normalized):\n")
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