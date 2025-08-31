import scipy.io
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv  # Cambio: Usar GAT en lugar de GCN
from torch.nn import GRU
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Configuración de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {device}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 1. Carga de datos
def load_data(mod_file, signal_files, cache_dir="cache_autoreg", max_neighbor_dist=0.2):
    print("Iniciando carga de datos...")
    mod_data = scipy.io.loadmat(mod_file)
    vecinos_raw = mod_data['Mod_Rect']['vecinos'][0, 0]
    vertices = mod_data['Mod_Rect']['vertices'][0, 0]
    N = vertices.shape[0]
    print(f"Number of nodes: {N}")
    print(f"Vertices shape: {vertices.shape}")
    print(f"Vertices range: x=[{np.min(vertices[:,0]):.2f}, {np.max(vertices[:,0]):.2f}], "
          f"y=[{np.min(vertices[:,1]):.2f}, {np.max(vertices[:,1]):.2f}], "
          f"z=[{np.min(vertices[:,2]):.2f}, {np.max(vertices[:,2]):.2f}] cm")

    # Rescale vertices (mm to cm) and translate to [0, 5] cm
    if np.max(vertices[:, 2]) > 1:  # Likely in mm
        print("Rescaling vertices from mm to cm...")
        vertices = vertices / 10.0
    print("Translating x, y to [0, 5] cm...")
    vertices[:, 0] += 2.5
    vertices[:, 1] += 2.5
    print(f"New vertices range: x=[{np.min(vertices[:,0]):.2f}, {np.max(vertices[:,0]):.2f}], "
          f"y=[{np.min(vertices[:,1]):.2f}, {np.max(vertices[:,1]):.2f}], "
          f"z=[{np.min(vertices[:,2]):.2f}, {np.max(vertices[:,2]):.2f}] cm")

    print("Construyendo edge_index...")
    edges = []
    for i in range(N):
        for j in vecinos_raw[i]:
            j = int(j) - 1 if j > 0 else j  # Adjust for 1-based indexing
            if 0 <= j < N and i != j:
                dist = np.linalg.norm(vertices[i] - vertices[j])
                if dist < max_neighbor_dist:
                    edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Mantener en CPU
    print(f"Máximo índice en edge_index: {edge_index.max().item()}")
    print(f"Total de aristas: {edge_index.shape[1]}")

    # Verificar nodos aislados
    degrees = torch.zeros(N, dtype=torch.long)
    for i in range(edge_index.shape[1]):
        degrees[edge_index[0, i]] += 1
        degrees[edge_index[1, i]] += 1
    print(f"Nodos aislados: {torch.sum(degrees == 0).item()}")

    print("Cargando señales de potencial de acción...")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_PA = os.path.join(cache_dir, "PA_concatenated.dat")
    n_time_steps = 1250 * len(signal_files)
    if os.path.exists(cache_PA):
        print("Cargando PA desde memmap...")
        PA = np.memmap(cache_PA, dtype='float32', mode='r', shape=(N, n_time_steps))
        mean = np.mean(PA, axis=(0, 1), keepdims=True)
        std = np.std(PA, axis=(0, 1), keepdims=True)
    else:
        print("Creando memmap para PA...")
        PA = np.memmap(cache_PA, dtype='float32', mode='w+', shape=(N, n_time_steps))
        offset = 0
        for signal_file in signal_files:
            print(f"Procesando {signal_file}...")
            with h5py.File(signal_file, 'r') as f:
                pa_data = np.array(f['PA'], dtype='float32').T
                PA[:, offset:offset+1250] = pa_data
                offset += 1250
        print("Normalizando PA...")
        mean = np.mean(PA, axis=(0, 1), keepdims=True)
        std = np.std(PA, axis=(0, 1), keepdims=True)
        PA[:] = (PA - mean) / (std + 1e-8)
        PA.flush()
    print(f"PA shape: {PA.shape}")
    return PA, edge_index, vertices, mean, std

# 2. Crear dataset autoregresivo
def create_autoregressive_dataset(PA, history=20, cache_dir="cache_autoreg"):
    print("Creando dataset autoregresivo...")
    n_nodes, n_time_steps = PA.shape
    n_snapshots = min(1000, n_time_steps - history)  # Aumentado para secuencia más larga
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_X = os.path.join(cache_dir, "X_autoregressive_small.dat")
    cache_Y = os.path.join(cache_dir, "Y_autoregressive_small.dat")
    if os.path.exists(cache_X) and os.path.exists(cache_Y):
        print("Cargando dataset desde memmap...")
        X = np.memmap(cache_X, dtype='float32', mode='r', shape=(n_snapshots, n_nodes, history))
        Y = np.memmap(cache_Y, dtype='float32', mode='r', shape=(n_snapshots, n_nodes))
    else:
        print("Creando memmap para X y Y...")
        X = np.memmap(cache_X, dtype='float32', mode='w+', shape=(n_snapshots, n_nodes, history))
        Y = np.memmap(cache_Y, dtype='float32', mode='w+', shape=(n_snapshots, n_nodes))
        for t in range(n_snapshots):
            if t % 10 == 0:
                print(f"Procesando snapshot {t}/{n_snapshots}")
            X[t] = PA[:, t:t+history]
            Y[t] = PA[:, t+history]
        X.flush()
        Y.flush()
    print(f"Dataset creado: X shape = {X.shape}, Y shape = {Y.shape}")
    return X, Y

# 3. Crear lista de Data
def create_data_list(X, Y, edge_index):
    print("Creando lista de Data objects...")
    data_list = []
    for i in range(X.shape[0]):
        if i % 10 == 0:
            print(f"Creando Data object {i}/{X.shape[0]}")
        x = torch.tensor(X[i], dtype=torch.float32)
        y = torch.tensor(Y[i], dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    print("Lista de Data objects creada.")
    return data_list

# 4. Modelo GAT con LSTM
class GAT_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, lstm_hidden, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.lstm = nn.LSTM(hidden_channels * heads, lstm_hidden, batch_first=True)  # LSTM en lugar de GRU
        self.conv2 = GATConv(lstm_hidden, 1, heads=1, dropout=0.3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()

# 5. Entrenamiento y evaluación
def train_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, loader, loss_fn, n_nodes):
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            loss = loss_fn(out, data.y)
            total_loss += loss.item()
            # Reshape to ensure (batch_size, n_nodes)
            batch_size = data.x.shape[0] // n_nodes
            predictions.append(out.cpu().view(batch_size, n_nodes).numpy())
            targets.append(data.y.cpu().view(batch_size, n_nodes).numpy())
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    print("Predictions shape:", predictions.shape)
    print("Targets shape:", targets.shape)

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    corr, _ = pearsonr(targets.flatten(), predictions.flatten())
    return total_loss / len(loader), rmse, mae, corr, predictions, targets

def evaluate_autoregressive(model, loader, loss_fn, n_nodes, edge_index, history=20, mean=None, std=None, model_dir = 'prueba'):
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    with torch.no_grad():
        # Convertir loader a lista para procesar snapshots secuencialmente
        data_list = list(loader)
        # Inicializar la ventana con el primer snapshot
        first_data = data_list[0]
        current_window = first_data.x.cpu().numpy()  # Forma: (n_nodes, history)
        
        for t, data in enumerate(data_list):
            data = data.to(device)
            # Convertir la ventana actual a tensor
            input_window = torch.tensor(current_window, dtype=torch.float32).to(device)
            # Usar el edge_index original, no el del batch
            edge_index = edge_index.to(device)
            out = model(input_window, edge_index)
            loss = loss_fn(out, data.y)
            total_loss += loss.item()
            
            # Guardar predicción y objetivo
            predictions.append(out.cpu().numpy())  # Forma: (n_nodes,)
            targets.append(data.y.cpu().numpy())   # Forma: (n_nodes,)
            
            # Actualizar ventana histórica con la predicción
            if t < len(data_list) - 1:  # Evitar actualizar en el último snapshot
                current_window = np.roll(current_window, -1, axis=1)  # Desplazar la ventana
                current_window[:, -1] = out.cpu().numpy()  # Añadir la predicción

    predictions = np.stack(predictions, axis=0)  # Forma: (n_snapshots, n_nodes)
    targets = np.stack(targets, axis=0)          # Forma: (n_snapshots, n_nodes)

    print("Autoregressive Predictions shape:", predictions.shape)
    print("Autoregressive Targets shape:", targets.shape)

    # Métricas en escala normalizada
    mse_normalized = mean_squared_error(targets, predictions)
    rmse_normalized = np.sqrt(mse_normalized)
    mae_normalized = mean_absolute_error(targets, predictions)
    corr = pearsonr(targets.flatten(), predictions.flatten())[0]
    r2_normalized = r2_score(targets.flatten(), predictions.flatten())
    ssim_values_normalized = []
    data_range_normalized = targets.max() - targets.min()
    per_step_mse_normalized = []
    per_step_ssim_normalized = []
    for t in range(predictions.shape[0]):
        mse_t = mean_squared_error(targets[t], predictions[t])
        ssim_t = structural_similarity(targets[t], predictions[t], data_range=data_range_normalized if data_range_normalized > 0 else 1)
        ssim_values_normalized.append(ssim_t)
        per_step_mse_normalized.append(mse_t)
        per_step_ssim_normalized.append(ssim_t)
    mean_ssim_normalized = np.mean(ssim_values_normalized)

    # Métricas en escala desnormalizada (original)
    mse_original = mse_normalized 
    rmse_original = rmse_normalized
    mae_original = mae_normalized
    r2_original = r2_normalized
    mean_ssim_original = mean_ssim_normalized
    per_step_mse_original = per_step_mse_normalized
    per_step_ssim_original = per_step_ssim_normalized
    if mean is not None and std is not None:
        predictions_orig = predictions * std + mean
        targets_orig = targets * std + mean
        mse_original = mean_squared_error(targets_orig, predictions_orig)
        rmse_original = np.sqrt(mse_original)
        mae_original = mean_absolute_error(targets_orig, predictions_orig)
        r2_original = r2_score(targets_orig.flatten(), predictions_orig.flatten())
        ssim_values_original = []
        data_range_original = targets_orig.max() - targets_orig.min()
        per_step_mse_original = []
        per_step_ssim_original = []
        for t in range(predictions.shape[0]):
            mse_t = mean_squared_error(targets_orig[t], predictions_orig[t])
            ssim_t = structural_similarity(targets_orig[t], predictions_orig[t], data_range=data_range_original if data_range_original > 0 else 1)
            ssim_values_original.append(ssim_t)
            per_step_mse_original.append(mse_t)
            per_step_ssim_original.append(ssim_t)
        mean_ssim_original = np.mean(ssim_values_original)
    
    metrics_file = os.path.join(model_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("\nMétricas en Escala Normalizada:\n")
        f.write(f"MSE: {mse_normalized:.4f}\n")
        f.write(f"RMSE: {rmse_normalized:.4f}\n")
        f.write(f"MAE: {mae_normalized:.4f}\n")
        f.write(f"Corr: {corr:.4f}\n")
        f.write(f"R²: {r2_normalized:.4f}\n")
        f.write(f"Mean SSIM: {mean_ssim_normalized:.4f}\n")

        f.write("\nMétricas en Escala Desnormalizada (Original):\n")
        f.write(f"MSE: {mse_original:.4f}\n")
        f.write(f"RMSE: {rmse_original:.4f}\n")
        f.write(f"MAE: {mae_original:.4f}\n")
        f.write(f"Corr: {corr:.4f}\n")
        f.write(f"R²: {r2_original:.4f}\n")
        f.write(f"Mean SSIM: {mean_ssim_original:.4f}")

    # Impresión de métricas
    print("\nMétricas en Escala Normalizada:")
    print(f"MSE: {mse_normalized:.4f}")
    print(f"RMSE: {rmse_normalized:.4f}")
    print(f"MAE: {mae_normalized:.4f}")
    print(f"Corr: {corr:.4f}")
    print(f"R²: {r2_normalized:.4f}")
    print(f"Mean SSIM: {mean_ssim_normalized:.4f}")

    print("\nMétricas en Escala Desnormalizada (Original):")
    print(f"MSE: {mse_original:.4f}")
    print(f"RMSE: {rmse_original:.4f}")
    print(f"MAE: {mae_original:.4f}")
    print(f"Corr: {corr:.4f}")
    print(f"R²: {r2_original:.4f}")
    print(f"Mean SSIM: {mean_ssim_original:.4f}")
    print(f"Metrics saved in: {metrics_file}")

    return total_loss / len(loader), rmse_normalized, mae_normalized, corr, predictions, targets, mse_original, mae_original, r2_original, mean_ssim_original, per_step_mse_original, per_step_ssim_original, per_step_mse_normalized, per_step_ssim_normalized

# 6. Visualización
def plot_predictions(predictions, targets, vertices, sample_nodes=[0, 100, 1000], filename="t_gcn_autoreg_predictions.png"):
    fig, axes = plt.subplots(len(sample_nodes), 1, figsize=(10, 5*len(sample_nodes)))
    if len(sample_nodes) == 1:
        axes = [axes]
    for i, node in enumerate(sample_nodes):
        axes[i].plot(targets[:, node], label="Ground Truth")
        axes[i].plot(predictions[:, node], label="Predicted")
        axes[i].set_title(f"Node {node}")
        axes[i].set_xlabel('Autoregressive step')
        axes[i].set_ylabel('Normalized PA')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_3d_predictions(predictions, vertices, t=0, filename="t_gcn_autoreg_3d_predictions.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=predictions[t], cmap='viridis')
    plt.colorbar(sc, label='Prediction')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    plt.savefig(filename)
    plt.close()

def create_comparative_3d_video(predictions, targets, vertices, output_file="comparative_3d_autoreg.gif", sample_nodes=2000, frame_interval=1):
    print("Preparando video comparativo 3D...")
    
    # Validar formas
    if predictions.shape != targets.shape:
        raise ValueError(f"Shapes mismatch: predictions {predictions.shape}, targets {targets.shape}")
    if predictions.shape[1] != vertices.shape[0]:
        raise ValueError(f"Number of nodes mismatch: predictions {predictions.shape[1]}, vertices {vertices.shape[0]}")
    
    # Submuestrear nodos
    if sample_nodes < vertices.shape[0]:
        idx = np.random.choice(vertices.shape[0], sample_nodes, replace=False)
        vertices = vertices[idx]
        predictions = predictions[:, idx]
        targets = targets[:, idx]
    
    # Calcular error absoluto
    errors = np.abs(predictions - targets)
    
    # Normalizar colores
    vmin = min(np.min(predictions), np.min(targets))
    vmax = max(np.max(predictions), np.max(targets))
    error_vmax = np.max(errors)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    def init():
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, 5)
        ax1.set_zlim(0, 0.3)
        ax1.set_box_aspect([5, 5, 0.3])
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.set_zlabel('Z (cm)')
        ax1.set_title('Predictions')

        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 5)
        ax2.set_zlim(0, 0.3)
        ax2.set_box_aspect([5, 5, 0.3])
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_zlabel('Z (cm)')
        ax2.set_title('Ground Truth')

        ax3.set_xlim(0, 5)
        ax3.set_ylim(0, 5)
        ax3.set_zlim(0, 0.3)
        ax3.set_box_aspect([5, 5, 0.3])
        ax3.set_xlabel('X (cm)')
        ax3.set_ylabel('Y (cm)')
        ax3.set_zlabel('Z (cm)')
        ax3.set_title('Absolute Error')
        return fig,

    def update(t):
        print(f"Rendering frame for t={t}")
        ax1.clear()
        ax2.clear()
        ax3.clear()

        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, 5)
        ax1.set_zlim(0, 0.3)
        ax1.set_box_aspect([5, 5, 0.3])
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.set_zlabel('Z (cm)')
        ax1.set_title(f'Predictions - Snapshot {t+85}')

        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 5)
        ax2.set_zlim(0, 0.3)
        ax2.set_box_aspect([5, 5, 0.3])
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_zlabel('Z (cm)')
        ax2.set_title(f'Ground Truth - Snapshot {t+85}')

        ax3.set_xlim(0, 5)
        ax3.set_ylim(0, 5)
        ax3.set_zlim(0, 0.3)
        ax3.set_box_aspect([5, 5, 0.3])
        ax3.set_xlabel('X (cm)')
        ax3.set_ylabel('Y (cm)')
        ax3.set_zlabel('Z (cm)')
        ax3.set_title(f'Absolute Error - Snapshot {t+85}')

        # Plot predictions
        sc1 = ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=predictions[t], cmap='viridis', s=20, vmin=vmin, vmax=vmax)
        # Plot ground truth
        sc2 = ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=targets[t], cmap='viridis', s=20, vmin=vmin, vmax=vmax)
        # Plot error
        sc3 = ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=errors[t], cmap='hot', s=20, vmin=0, vmax=error_vmax)

        # Añadir barras de color solo en el primer frame
        if t == 0:
            plt.colorbar(sc1, ax=ax1, label='Prediction (Normalized PA)')
            plt.colorbar(sc2, ax=ax2, label='Ground Truth (Normalized PA)')
            plt.colorbar(sc3, ax=ax3, label='Absolute Error')
        return fig,

    # Crear animación
    n_frames = len(predictions) // frame_interval
    frames = range(0, len(predictions), frame_interval)
    anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=200, blit=False)

    # Guardar como GIF
    print(f"Saving GIF to {output_file}...")
    writer = PillowWriter(fps=5)
    anim.save(output_file, writer=writer)
    plt.close()
    print("GIF generated successfully.")
    return output_file

# 7. Pipeline principal
def main():
    mod_file = "Mod_Rect.mat"
    signal_files = ["Rect_dif_25_remod_50_1.mat", "Rect_dif_25_remod_50_2.mat"]
    PA, edge_index, vertices, mean, std = load_data(mod_file, signal_files, cache_dir="cache_autoreg", max_neighbor_dist=0.2)
    n_nodes = PA.shape[0]  # 63072 nodes
    
    X, Y = create_autoregressive_dataset(PA, history=20, cache_dir="cache_autoreg")
    
    data_list = create_data_list(X, Y, edge_index)
    train_data = data_list[:70]
    val_data = data_list[70:85]
    test_data = data_list[85:]
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)  # batch_size=1 para test
    
    model = GAT_LSTM(in_channels=20, hidden_channels=16, lstm_hidden=32, heads=4).to(device)  # Ajuste: hidden reducido por heads
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    model_dir = f"resultados_{script_name}"
    os.makedirs(model_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    print("\nEntrenando GAT-LSTM...")
    best_val_loss = float('inf')
    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_rmse, val_mae, val_corr, _, _ = evaluate(model, val_loader, loss_fn, n_nodes)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "gat_lstm_autoreg_best.pt"))

    # Guardar gráfica de pérdidas
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "loss_curve.png"))
    plt.close()

    # Evaluación en el conjunto de prueba con predicciones autoregresivas
    model.load_state_dict(torch.load(os.path.join(model_dir, "gat_lstm_autoreg_best.pt")))
    test_loss, test_rmse, test_mae, test_corr, predictions, targets, mse_global, mae_global, r2_global, mean_ssim, test_per_step_mse_original, test_per_step_ssim_original, test_per_step_mse_normalized, test_per_step_ssim_normalized = evaluate_autoregressive(
        model, test_loader, loss_fn, n_nodes, edge_index, history=20, mean=mean, std=std, model_dir = model_dir
    )
    print(f"GAT-LSTM Autoregressive: MSE = {test_loss:.6f}, RMSE = {test_rmse:.6f}, MAE = {test_mae:.6f}, Corr = {test_corr:.6f}")

    # Generar visualizaciones
    plot_predictions(predictions, targets, vertices, filename=os.path.join(model_dir, "gat_lstm_autoreg_predictions.png"))

    # Nuevas gráficas: MSE y SSIM a lo largo del tiempo (para la secuencia de test)
    plt.figure(figsize=(10, 6))
    plt.plot(test_per_step_mse_normalized, label='MSE per step')
    plt.xlabel('Autoregressive step')
    plt.ylabel('MSE (normalized scale)')
    plt.title('MSE over autoregressive steps (full test)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "mse_over_time.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(test_per_step_ssim_normalized, label='SSIM per step')
    plt.xlabel('Autoregressive step')
    plt.ylabel('SSIM (normalized scale)')
    plt.title('SSIM over autoregressive steps (full test)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "ssim_over_time.png"))
    plt.close()

    plot_3d_predictions(predictions, vertices, t=0, filename=os.path.join(model_dir, "gat_lstm_autoreg_3d_predictions.png"))
    
    # Generar video comparativo
    frame_interval = max(1, len(predictions) // 100)
    comparative_gif = create_comparative_3d_video(
        predictions, 
        targets, 
        vertices, 
        output_file=os.path.join(model_dir, "comparative_3d_autoreg.gif"),
        sample_nodes=2000,
        frame_interval=frame_interval
    )
    print(f"Video comparativo guardado en: {comparative_gif}")

if __name__ == "__main__":
    main()