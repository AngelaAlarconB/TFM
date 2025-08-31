import scipy.io
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
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
def load_data(file_path, cache_dir="cache_structured", max_neighbor_dist=0.2,
              convert_mm_to_cm=True, target_range=7.0, uniform_scale=True, exclude_time_steps=300):
    print("Loading data...")
    data = scipy.io.loadmat(file_path)
    PA = data['PA']
    mod = data['mod'][0, 0]
    vecinos_raw = mod['vecinos']
    vertices = mod['vertices'].astype(np.float32)

    N_original = vertices.shape[0]
    n_time_steps = PA.shape[1]
    print(f"Original PA shape: {PA.shape}")
    print(f"Original vertices shape: {vertices.shape}")

    if exclude_time_steps > 0:
        print(f"Excluding first {exclude_time_steps} time steps...")
        PA = PA[:, exclude_time_steps:]
        n_time_steps = PA.shape[1]
        print(f"PA shape after time step exclusion: {PA.shape}")

    try:
        vertices_reshaped = vertices.reshape(232, 232, 2, 3)
        PA_reshaped = PA.reshape(232, 232, 2, n_time_steps)
    except ValueError as e:
        raise ValueError(f"Error reshaping data: {e}. Check dimensions.")

    print("Restricting grid to [:230, :230, :]...")
    vertices_reshaped = vertices_reshaped[:230, :230, :, :]
    PA_reshaped = PA_reshaped[:230, :230, :, :]
    N = 230 * 230 * 2
    vertices = vertices_reshaped.reshape(N, 3)
    PA = PA_reshaped.reshape(N, n_time_steps)
    print(f"New PA shape: {PA.shape}")
    print(f"New vertices shape: {vertices.shape}")

    print("Adjusting neighbor indices for restricted grid...")
    node_map = np.full(N_original, -1, dtype=int)
    for i in range(230):
        for j in range(230):
            for k in range(2):
                old_idx = (k * 232 * 232) + (j * 232) + i
                new_idx = (k * 230 * 230) + (j * 230) + i
                node_map[old_idx] = new_idx
    vecinos_new = []
    for i in range(N_original):
        if node_map[i] != -1:
            neigh = np.array(vecinos_raw[i]).flatten()
            new_neigh = [node_map[int(val) - 1] for val in neigh if int(val) - 1 < N_original and node_map[int(val) - 1] != -1]
            vecinos_new.append(new_neigh)
        else:
            vecinos_new.append([])
    vecinos_raw = vecinos_new[:N]

    print(f"X range before scaling: [{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}]")
    print(f"Y range before scaling: [{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")
    print(f"Z range before scaling: [{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")

    if convert_mm_to_cm and vertices[:,2].max() > 1.0:
        print("Converting vertices from mm to cm (dividing by 10).")
        vertices /= 10.0

    xmin, xmax = vertices[:,0].min(), vertices[:,0].max()
    ymin, ymax = vertices[:,1].min(), vertices[:,1].max()
    range_x = xmax - xmin
    range_y = ymax - ymin
    if range_x <= 0 or range_y <= 0:
        raise ValueError("Ranges for x or y are zero; check vertices.")

    if uniform_scale:
        max_range = max(range_x, range_y)
        scale = target_range / max_range
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        vertices[:,0] = (vertices[:,0] - cx) * scale + target_range / 2.0
        vertices[:,1] = (vertices[:,1] - cy) * scale + target_range / 2.0
    else:
        vertices[:,0] = (vertices[:,0] - xmin) / range_x * target_range
        vertices[:,1] = (vertices[:,1] - ymin) / range_y * target_range

    print(f"Scaled x range: [{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}]")
    print(f"Scaled y range: [{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")

    edges = []
    for i in range(N):
        neigh = np.array(vecinos_raw[i]).flatten()
        for val in neigh:
            try:
                j = int(val)
            except Exception:
                continue
            if 0 <= j < N and i != j:
                dist = np.linalg.norm(vertices[i, :2] - vertices[j, :2])
                if dist < max_neighbor_dist:
                    edges.append([i, j])
    edge_index = np.array(edges).T if len(edges) > 0 else np.zeros((2, 0), dtype=int)
    print(f"Edge index shape after filtering: {edge_index.shape}")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_PA = os.path.join(cache_dir, f"PA_concatenated_N{N}_T{n_time_steps}.dat")
    mean_file = os.path.join(cache_dir, "PA_mean.npy")
    std_file = os.path.join(cache_dir, "PA_std.npy")
    expected_bytes = N * n_time_steps * np.dtype('float32').itemsize

    if os.path.exists(cache_PA):
        actual_bytes = os.path.getsize(cache_PA)
        if actual_bytes != expected_bytes:
            print("Cache memmap size mismatch: recreating memmap with new shape.")
            os.remove(cache_PA)
            create_new_memmap = True
        else:
            create_new_memmap = False
    else:
        create_new_memmap = True

    if create_new_memmap:
        print("Creating memmap for PA and normalizing (global).")
        PA_mem = np.memmap(cache_PA, dtype='float32', mode='w+', shape=(N, n_time_steps))
        PA_mem[:] = PA.astype('float32')
        mean = np.mean(PA_mem)
        std = np.std(PA_mem)
        PA_mem[:] = (PA_mem - mean) / (std + 1e-8)
        PA_mem.flush()
        PA = PA_mem
        np.save(mean_file, mean)
        np.save(std_file, std)
    else:
        print("Loading PA from existing memmap.")
        PA = np.memmap(cache_PA, dtype='float32', mode='r', shape=(N, n_time_steps))
        mean = np.load(mean_file) if os.path.exists(mean_file) else np.mean(PA)
        std = np.load(std_file) if os.path.exists(std_file) else np.std(PA)

    print(f"Final PA shape: {PA.shape}, N nodes: {N}")
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    return PA, edge_index, vertices, vecinos_raw, mean, std

# 2. Crear dataset autoregresivo
def create_autoregressive_dataset(PA, history=20, cache_dir="cache_structured"):
    print("Creando dataset autoregresivo...")
    n_nodes, n_time_steps = PA.shape
    n_snapshots = n_time_steps - history
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_X = os.path.join(cache_dir, "X_autoregressive.dat")
    cache_Y = os.path.join(cache_dir, "Y_autoregressive.dat")
    
    expected_X_bytes = n_snapshots * n_nodes * history * np.dtype('float32').itemsize
    expected_Y_bytes = n_snapshots * n_nodes * np.dtype('float32').itemsize
    
    create_new_memmap = False
    if os.path.exists(cache_X) and os.path.exists(cache_Y):
        actual_X_bytes = os.path.getsize(cache_X)
        actual_Y_bytes = os.path.getsize(cache_Y)
        if actual_X_bytes != expected_X_bytes or actual_Y_bytes != expected_Y_bytes:
            print("Cache memmap size mismatch: recreating memmap files.")
            os.remove(cache_X)
            os.remove(cache_Y)
            create_new_memmap = True
        else:
            create_new_memmap = False
    else:
        create_new_memmap = True
    
    if create_new_memmap:
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
    else:
        print("Cargando dataset desde memmap...")
        X = np.memmap(cache_X, dtype='float32', mode='r', shape=(n_snapshots, n_nodes, history))
        Y = np.memmap(cache_Y, dtype='float32', mode='r', shape=(n_snapshots, n_nodes))
    
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

# 4. Modelo T-GCN
class TGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, gru_hidden):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.gru = GRU(hidden_channels, gru_hidden, batch_first=True)
        self.conv2 = GCNConv(gru_hidden, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = x.unsqueeze(0)
        x, _ = self.gru(x)
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

def evaluate_autoregressive(model, loader, loss_fn, n_nodes, edge_index, history=20, mean=None, std=None):
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    with torch.no_grad():
        data_list = list(loader)
        first_data = data_list[0]
        current_window = first_data.x.cpu().numpy()
        
        for t, data in enumerate(data_list):
            data = data.to(device)
            input_window = torch.tensor(current_window, dtype=torch.float32).to(device)
            edge_index = edge_index.to(device)
            print(f"Snapshot {t}: edge_index max={edge_index.max().item()}, min={edge_index.min().item()}")
            out = model(input_window, edge_index)
            loss = loss_fn(out, data.y)
            total_loss += loss.item()
            
            predictions.append(out.cpu().numpy())
            targets.append(data.y.cpu().numpy())
            
            if t < len(data_list) - 1:
                current_window = np.roll(current_window, -1, axis=1)
                current_window[:, -1] = out.cpu().numpy()

    predictions = np.stack(predictions, axis=0)
    targets = np.stack(targets, axis=0)

    print("Autoregressive Predictions shape:", predictions.shape)
    print("Autoregressive Targets shape:", targets.shape)

    mse_normalized = mean_squared_error(targets, predictions)
    rmse_normalized = np.sqrt(mse_normalized)
    mae_normalized = mean_absolute_error(targets, predictions)
    corr = pearsonr(targets.flatten(), predictions.flatten())[0]
    r2_normalized = r2_score(targets.flatten(), predictions.flatten())
    data_range_normalized = targets.max() - targets.min()
    per_step_mse_normalized = []
    for t in range(predictions.shape[0]):
        mse_t = mean_squared_error(targets[t], predictions[t])
        per_step_mse_normalized.append(mse_t)

    mse_global = mse_normalized
    mae_global = mae_normalized
    r2_global = r2_normalized
    per_step_mse = per_step_mse_normalized

    if mean is not None and std is not None:
        predictions_orig = predictions * std + mean
        targets_orig = targets * std + mean
        mse_global = mean_squared_error(targets_orig, predictions_orig)
        mae_global = mean_absolute_error(targets_orig, predictions_orig)
        r2_global = r2_score(targets_orig.flatten(), predictions_orig.flatten())
        data_range = targets_orig.max() - targets_orig.min()
        per_step_mse = []
        for t in range(predictions.shape[0]):
            mse_t = mean_squared_error(targets_orig[t], predictions_orig[t])
            per_step_mse.append(mse_t)

    print(f"Metrics in normalized scale:")
    print(f"Global MSE: {mse_normalized:.4f}")
    print(f"Global MAE: {mae_normalized:.4f}")
    print(f"Global R²: {r2_normalized:.4f}")

    print(f"Metrics in original scale:")
    print(f"Global MSE: {mse_global:.4f}")
    print(f"Global MAE: {mae_global:.4f}")
    print(f"Global R²: {r2_global:.4f}")

    return total_loss / len(loader), rmse_normalized, mae_normalized, corr, predictions, targets, mse_global, mae_global, r2_global, per_step_mse, per_step_mse_normalized

# 6. Visualización
def plot_predictions(predictions, targets, vertices, sample_nodes=[0, 100, 1000], filename="t_gcn_structured_predictions.png", time_offset=405):
    fig, axes = plt.subplots(len(sample_nodes), 1, figsize=(10, 5*len(sample_nodes)))
    if len(sample_nodes) == 1:
        axes = [axes]
    for i, node in enumerate(sample_nodes):
        axes[i].plot(range(time_offset, time_offset + len(targets)), targets[:, node], label="Ground Truth")
        axes[i].plot(range(time_offset, time_offset + len(predictions)), predictions[:, node], label="Predicted")
        axes[i].set_title(f"Node {node} (t={time_offset} to t={time_offset + len(targets) - 1})")
        axes[i].set_xlabel('Original Time Step')
        axes[i].set_ylabel('Normalized PA')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_3d_predictions(predictions, vertices, t=0, filename="t_gcn_structured_3d_predictions.png", time_offset=405):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=predictions[t], cmap='viridis')
    plt.colorbar(sc, label='Prediction (Normalized PA)')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax.set_title(f'3D Predictions - Snapshot t={time_offset + t}')
    plt.savefig(filename)
    plt.close()

def create_comparative_3d_video(predictions, targets, vertices, output_file="comparative_3d_structured.gif", sample_nodes=2000, frame_interval=1, time_offset=1100):
    print("Preparing 3D comparative video...")
    
    if predictions.shape != targets.shape:
        raise ValueError(f"Shapes mismatch: predictions {predictions.shape}, targets {targets.shape}")
    if predictions.shape[1] != vertices.shape[0]:
        raise ValueError(f"Number of nodes mismatch: predictions {predictions.shape[1]}, vertices {vertices.shape[0]}")
    
    if sample_nodes < vertices.shape[0]:
        idx = np.random.choice(vertices.shape[0], sample_nodes, replace=False)
        vertices = vertices[idx]
        predictions = predictions[:, idx]
        targets = targets[:, idx]
    
    errors = np.abs(predictions - targets)
    
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
        ax1.set_xlim(0, 7)
        ax1.set_ylim(0, 7)
        ax1.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        ax1.set_box_aspect([7, 7, vertices[:, 2].max() - vertices[:, 2].min()])
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.set_zlabel('Z (cm)')
        ax1.set_title('Predictions')

        ax2.set_xlim(0, 7)
        ax2.set_ylim(0, 7)
        ax2.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        ax2.set_box_aspect([7, 7, vertices[:, 2].max() - vertices[:, 2].min()])
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_zlabel('Z (cm)')
        ax2.set_title('Ground Truth')

        ax3.set_xlim(0, 7)
        ax3.set_ylim(0, 7)
        ax3.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        ax3.set_box_aspect([7, 7, vertices[:, 2].max() - vertices[:, 2].min()])
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

        ax1.set_xlim(0, 7)
        ax1.set_ylim(0, 7)
        ax1.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        ax1.set_box_aspect([7, 7, vertices[:, 2].max() - vertices[:, 2].min()])
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.set_zlabel('Z (cm)')
        ax1.set_title(f'Predictions - Snapshot t={time_offset + t}')

        ax2.set_xlim(0, 7)
        ax2.set_ylim(0, 7)
        ax2.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        ax2.set_box_aspect([7, 7, vertices[:, 2].max() - vertices[:, 2].min()])
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_zlabel('Z (cm)')
        ax2.set_title(f'Ground Truth - Snapshot t={time_offset + t}')

        ax3.set_xlim(0, 7)
        ax3.set_ylim(0, 7)
        ax3.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        ax3.set_box_aspect([7, 7, vertices[:, 2].max() - vertices[:, 2].min()])
        ax3.set_xlabel('X (cm)')
        ax3.set_ylabel('Y (cm)')
        ax3.set_zlabel('Z (cm)')
        ax3.set_title(f'Absolute Error - Snapshot t={time_offset + t}')

        sc1 = ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=predictions[t], cmap='viridis', s=20, vmin=vmin, vmax=vmax)
        sc2 = ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=targets[t], cmap='viridis', s=20, vmin=vmin, vmax=vmax)
        sc3 = ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=errors[t], cmap='hot', s=20, vmin=0, vmax=error_vmax)

        if t == 0:
            plt.colorbar(sc1, ax=ax1, label='Prediction (Normalized)')
            plt.colorbar(sc2, ax=ax2, label='Ground Truth (Normalized)')
            plt.colorbar(sc3, ax=ax3, label='Absolute Error')
        return fig,

    n_frames = len(predictions) // frame_interval
    frames = range(0, len(predictions), frame_interval)
    anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=200, blit=False)

    print(f"Saving GIF to {output_file}...")
    writer = PillowWriter(fps=5)
    anim.save(output_file, writer=writer)
    plt.close()
    print("GIF generated successfully.")
    return output_file

# 7. Pipeline principal
def main():
    file_path = "datos/RotorSquare_Remod100.mat"
    PA, edge_index, vertices, _, mean, std = load_data(file_path, max_neighbor_dist=0.2, cache_dir="cache_structured", exclude_time_steps=300)
    n_nodes = PA.shape[0]
    
    if PA.shape[1] < 901:  # Ajustado para cubrir hasta PA[:,900] (t=1200)
        raise ValueError(f"PA has {PA.shape[1]} time steps; need at least 901 for t=1100 to t=1200")
    
    X, Y = create_autoregressive_dataset(PA, history=20, cache_dir="cache_structured")
    data_list = create_data_list(X, Y, edge_index)
    train_data = data_list[:70]
    val_data = data_list[70:85]
    test_data = data_list[780:]  # All remaining for full test sequence
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)
    
    model = TGCN(in_channels=20, hidden_channels=64, gru_hidden=32).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    model_dir = f"resultados_{script_name}"
    os.makedirs(model_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    print("\nEntrenando T-GCN...")
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
            torch.save(model.state_dict(), os.path.join(model_dir, "t_gcn_structured_best.pt"))

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

    model.load_state_dict(torch.load(os.path.join(model_dir, "t_gcn_structured_best.pt")))
    test_loss, test_rmse, test_mae, test_corr, predictions, targets, mse_global, mae_global, r2_global, test_per_step_mse, test_per_step_mse_normalized = evaluate_autoregressive(
        model, test_loader, loss_fn, n_nodes, edge_index, history=20, mean=mean, std=std
    )
    print(f"T-GCN Autoregressive: MSE = {test_loss:.6f}, RMSE = {test_rmse:.6f}, MAE = {test_mae:.6f}, Corr = {test_corr:.6f}")

    start_time = 1100
    test_length = len(test_data)
    end_time = start_time + test_length - 1

    metrics_file = os.path.join(model_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Metrics in normalized scale:\n")
        f.write(f"MSE: {test_loss:.6f}\n")
        f.write(f"RMSE: {test_rmse:.6f}\n")
        f.write(f"MAE: {test_mae:.6f}\n")
        f.write(f"Correlation: {test_corr:.6f}\n")
        f.write(f"\nMetrics in original scale:\n")
        f.write(f"Global MSE: {mse_global:.4f}\n")
        f.write(f"Global MAE: {mae_global:.4f}\n")
        f.write(f"Global R²: {r2_global:.4f}\n")
    print(f"Metrics saved in: {metrics_file}")

    plot_predictions(predictions, targets, vertices, filename=os.path.join(model_dir, "t_gcn_structured_predictions.png"), time_offset=start_time)
    plot_3d_predictions(predictions, vertices, t=0, filename=os.path.join(model_dir, "t_gcn_structured_3d_predictions.png"), time_offset=start_time)

    # Gráficas: MSE a lo largo del tiempo para test
    plt.figure(figsize=(10, 6))
    plt.plot(test_per_step_mse_normalized, label='MSE per step')
    plt.xlabel('Autoregressive step')
    plt.ylabel('MSE (normalized scale)')
    plt.title(f'MSE over autoregressive steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "mse_over_time.png"))
    plt.close()

    # Gráfica de predicción vs. real para los 3 nodos en la secuencia de test
    plot_predictions(predictions, targets, vertices, sample_nodes=[0, 100, 1000], filename=os.path.join(model_dir, "test_predictions_nodes.png"), time_offset=start_time)

    comparative_gif = create_comparative_3d_video(
        predictions, 
        targets, 
        vertices, 
        output_file=os.path.join(model_dir, f"comparative_3d_structured_t{start_time}_t{end_time}.gif"),
        sample_nodes=2000,
        frame_interval=1,
        time_offset=start_time
    )
    print(f"Comparative video saved in: {comparative_gif}")

if __name__ == "__main__":
    main()