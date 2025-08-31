import scipy.io
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
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
    if not edges:
        raise ValueError("No edges created. Check max_neighbor_dist or vecinos_raw data.")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Validate edge_index
    if edge_index.max().item() >= N or edge_index.min().item() < 0:
        raise ValueError(f"edge_index contains invalid indices: max={edge_index.max().item()}, min={edge_index.min().item()}, expected [0, {N-1}]")
    print(f"Máximo índice en edge_index: {edge_index.max().item()}")
    print(f"Total de aristas: {edge_index.shape[1]}")

    # Verificar nodos aislados
    degrees = torch.zeros(N, dtype=torch.long)
    for i in range(edge_index.shape[1]):
        degrees[edge_index[0, i]] += 1
        degrees[edge_index[1, i]] += 1
    isolated_nodes = torch.sum(degrees == 0).item()
    print(f"Nodos aislados: {isolated_nodes}")
    if isolated_nodes > 0:
        print(f"Warning: {isolated_nodes} isolated nodes detected. Self-loops will be added in GCNConv.")

    print("Cargando señales de potencial de acción...")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_PA = os.path.join(cache_dir, "PA_concatenated.dat")
    n_time_steps = 1250 * len(signal_files)
    if os.path.exists(cache_PA):
        print("Cargando PA desde memmap...")
        PA = np.memmap(cache_PA, dtype='float32', mode='r', shape=(N, n_time_steps))
        mean_file = os.path.join(cache_dir, "PA_mean.npy")
        std_file = os.path.join(cache_dir, "PA_std.npy")
        if os.path.exists(mean_file) and os.path.exists(std_file):
            mean = np.load(mean_file)
            std = np.load(std_file)
        else:
            mean = np.mean(PA, axis=(0, 1), keepdims=True)
            std = np.std(PA, axis=(0, 1), keepdims=True)
            np.save(mean_file, mean)
            np.save(std_file, std)
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
        np.save(os.path.join(cache_dir, "PA_mean.npy"), mean)
        np.save(os.path.join(cache_dir, "PA_std.npy"), std)
    print(f"PA shape: {PA.shape}")
    return PA, edge_index, vertices, mean, std

# 2. Crear dataset autoregresivo
def create_autoregressive_dataset(PA, history=20, cache_dir="cache_autoreg"):
    print("Creando dataset autoregresivo...")
    n_nodes, n_time_steps = PA.shape
    n_snapshots = min(100, n_time_steps - history)
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

# 4. Modelo
class GCRN(nn.Module):
    def __init__(self, in_channels, hidden_channels, gru_hidden):
        super().__init__()
        self.gcn_in = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_channels, gru_hidden, batch_first=True)
        self.gcn_out = GCNConv(gru_hidden, 1, add_self_loops=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_seq, edge_index, batch=None):
        # x_seq: [batch_size * n_nodes, history] or [batch_size, n_nodes, history]
        # edge_index: [2, num_edges], adjusted for batch
        # batch: [batch_size * n_nodes], indicates which graph each node belongs to
        if x_seq.dim() == 3:  # [batch_size, n_nodes, history]
            batch_size, n_nodes, history = x_seq.shape
            x_seq = x_seq.view(batch_size * n_nodes, history)  # [batch_size * n_nodes, history]
        else:
            batch_size = 1 if batch is None else batch.max().item() + 1
            n_nodes = x_seq.shape[0] if batch is None else x_seq.shape[0] // batch_size
            history = x_seq.shape[1]

        # Process each time step with GCN
        h_seq = []
        for t in range(history):
            x_t = x_seq[:, t].unsqueeze(-1)  # [batch_size * n_nodes, 1]
            x_t = self.gcn_in(x_t, edge_index)  # [batch_size * n_nodes, hidden_channels]
            x_t = self.relu(x_t)
            h_seq.append(x_t)

        h_seq = torch.stack(h_seq, dim=1)  # [batch_size * n_nodes, history, hidden_channels]
        # Reshape for GRU: process each node's sequence independently
        h_seq = h_seq.view(batch_size, n_nodes, history, -1)  # [batch_size, n_nodes, history, hidden_channels]
        h_seq = h_seq.permute(0, 1, 2, 3)  # [batch_size, n_nodes, history, hidden_channels]
        # Flatten batch and node dimensions for GRU
        h_seq = h_seq.reshape(batch_size * n_nodes, history, -1)  # [batch_size * n_nodes, history, hidden_channels]
        out_seq, _ = self.gru(h_seq)  # [batch_size * n_nodes, history, gru_hidden]
        out_last = out_seq[:, -1, :]  # [batch_size * n_nodes, gru_hidden]
        out_last = self.dropout(out_last)
        out = self.gcn_out(out_last, edge_index)  # [batch_size * n_nodes, 1]
        return out.view(batch_size, n_nodes)  # [batch_size, n_nodes]

# 5. Entrenamiento y evaluación
def train_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y.view(out.shape))
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
            out = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(out, data.y.view(out.shape))
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
        current_window = first_data.x.cpu().numpy()  # [n_nodes, history]
        
        for t, data in enumerate(data_list):
            data = data.to(device)
            edge_index = edge_index.to(device)  # Use original edge_index
            print(f"Snapshot {t}: edge_index max={edge_index.max().item()}, min={edge_index.min().item()}")
            if edge_index.max().item() >= n_nodes or edge_index.min().item() < 0:
                raise ValueError(f"Invalid edge_index in autoregressive eval: max={edge_index.max().item()}, min={edge_index.min().item()}")
            input_window = torch.tensor(current_window, dtype=torch.float32).to(device)  # [n_nodes, history]
            out = model(input_window, edge_index)  # [1, n_nodes]
            loss = loss_fn(out, data.y)
            total_loss += loss.item()
            
            predictions.append(out.squeeze(0).cpu().numpy())  # [1, n_nodes]
            targets.append(data.y.cpu().numpy())   # [n_nodes]
            
            if t < len(data_list) - 1:
                current_window = np.roll(current_window, -1, axis=1)
                current_window[:, -1] = out.cpu().numpy().flatten()  # Ensure shape matches

    predictions = np.stack(predictions, axis=0)  # [n_snapshots, n_nodes]
    targets = np.stack(targets, axis=0)          # [n_snapshots, n_nodes]

    print("Autoregressive Predictions shape:", predictions.shape)
    print("Autoregressive Targets shape:", targets.shape)

    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    corr, _ = pearsonr(targets.flatten(), predictions.flatten())

    mse_global = mse
    mae_global = mae
    r2_global = 0.0
    mean_ssim = 0.0
    if mean is not None and std is not None:
        predictions_orig = predictions * std + mean
        targets_orig = targets * std + mean
        mse_global = mean_squared_error(targets_orig, predictions_orig)
        mae_global = mean_absolute_error(targets_orig, predictions_orig)
        r2_global = r2_score(targets_orig.flatten(), predictions_orig.flatten())
        ssim_values = []
        data_range = targets_orig.max() - targets_orig.min()
        for t in range(predictions.shape[0]):
            ssim_t = structural_similarity(targets_orig[t], predictions_orig[t], data_range=data_range)
            ssim_values.append(ssim_t)
        mean_ssim = np.mean(ssim_values)

    print(f"Métricas en escala original:")
    print(f"MSE global: {mse_global:.4f}")
    print(f"MAE global: {mae_global:.4f}")
    print(f"R² global: {r2_global:.4f}")
    print(f"SSIM medio: {mean_ssim:.4f}")

    return total_loss / len(loader), rmse, mae, corr, predictions, targets, mse_global, mae_global, r2_global, mean_ssim

# 6. Visualización
def plot_predictions(predictions, targets, vertices, sample_nodes=[0, 100, 1000], filename="t_gcn_autoreg_predictions.png"):
    fig, axes = plt.subplots(len(sample_nodes), 1, figsize=(10, 5*len(sample_nodes)))
    if len(sample_nodes) == 1:
        axes = [axes]
    for i, node in enumerate(sample_nodes):
        axes[i].plot(targets[:, node], label="Real")
        axes[i].plot(predictions[:, node], label="Predicho")
        axes[i].set_title(f"Nodo {node}")
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
        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, 5)
        ax1.set_zlim(0, 0.3)
        ax1.set_box_aspect([5, 5, 0.3])
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.set_zlabel('Z (cm)')
        ax1.set_title('Predicciones')

        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 5)
        ax2.set_zlim(0, 0.3)
        ax2.set_box_aspect([5, 5, 0.3])
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_zlabel('Z (cm)')
        ax2.set_title('Reales')

        ax3.set_xlim(0, 5)
        ax3.set_ylim(0, 5)
        ax3.set_zlim(0, 0.3)
        ax3.set_box_aspect([5, 5, 0.3])
        ax3.set_xlabel('X (cm)')
        ax3.set_ylabel('Y (cm)')
        ax3.set_zlabel('Z (cm)')
        ax3.set_title('Error Absoluto')
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
        ax1.set_title(f'Predicciones - Snapshot {t+85}')

        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 5)
        ax2.set_zlim(0, 0.3)
        ax2.set_box_aspect([5, 5, 0.3])
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_zlabel('Z (cm)')
        ax2.set_title(f'Reales - Snapshot {t+85}')

        ax3.set_xlim(0, 5)
        ax3.set_ylim(0, 5)
        ax3.set_zlim(0, 0.3)
        ax3.set_box_aspect([5, 5, 0.3])
        ax3.set_xlabel('X (cm)')
        ax3.set_ylabel('Y (cm)')
        ax3.set_zlabel('Z (cm)')
        ax3.set_title(f'Error Absoluto - Snapshot {t+85}')

        sc1 = ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=predictions[t], cmap='viridis', s=20, vmin=vmin, vmax=vmax)
        sc2 = ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=targets[t], cmap='viridis', s=20, vmin=vmin, vmax=vmax)
        sc3 = ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=errors[t], cmap='hot', s=20, vmin=0, vmax=error_vmax)

        if t == 0:
            plt.colorbar(sc1, ax=ax1, label='Predicción (PA normalizado)')
            plt.colorbar(sc2, ax=ax2, label='Real (PA normalizado)')
            plt.colorbar(sc3, ax=ax3, label='Error Absoluto')
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
    mod_file = "Mod_Rect.mat"
    signal_files = ["Rect_dif_25_remod_50_1.mat", "Rect_dif_25_remod_50_2.mat"]
    PA, edge_index, vertices, mean, std = load_data(mod_file, signal_files, cache_dir="cache_autoreg", max_neighbor_dist=0.2)
    n_nodes = PA.shape[0]
    
    X, Y = create_autoregressive_dataset(PA, history=20, cache_dir="cache_autoreg")
    
    data_list = create_data_list(X, Y, edge_index)
    train_data = data_list[:70]
    val_data = data_list[70:85]
    test_data = data_list[85:]
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)
    
    model = GCRN(in_channels=1, hidden_channels=64, gru_hidden=32).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    model_dir = f"resultados_{script_name}"
    os.makedirs(model_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
            torch.save(model.state_dict(), os.path.join(model_dir, "t_gcn_autoreg_best.pt"))

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

    model.load_state_dict(torch.load(os.path.join(model_dir, "t_gcn_autoreg_best.pt")))
    test_loss, test_rmse, test_mae, test_corr, predictions, targets, mse_global, mae_global, r2_global, mean_ssim = evaluate_autoregressive(
        model, test_loader, loss_fn, n_nodes, edge_index, history=20, mean=mean, std=std
    )
    print(f"T-GCN Autoregressive: MSE = {test_loss:.6f}, RMSE = {test_rmse:.6f}, MAE = {test_mae:.6f}, Corr = {test_corr:.6f}")

    metrics_file = os.path.join(model_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Métricas en escala normalizada:\n")
        f.write(f"MSE: {test_loss:.6f}\n")
        f.write(f"RMSE: {test_rmse:.6f}\n")
        f.write(f"MAE: {test_mae:.6f}\n")
        f.write(f"Correlación: {test_corr:.6f}\n")
        f.write("\nMétricas en escala original:\n")
        f.write(f"MSE global: {mse_global:.4f}\n")
        f.write(f"MAE global: {mae_global:.4f}\n")
        f.write(f"R² global: {r2_global:.4f}\n")
        f.write(f"SSIM medio: {mean_ssim:.4f}\n")
    print(f"Métricas guardadas en: {metrics_file}")

    plot_predictions(predictions, targets, vertices, filename=os.path.join(model_dir, "t_gcn_autoreg_predictions.png"))
    plot_3d_predictions(predictions, vertices, t=0, filename=os.path.join(model_dir, "t_gcn_autoreg_3d_predictions.png"))
    
    comparative_gif = create_comparative_3d_video(
        predictions, 
        targets, 
        vertices, 
        output_file=os.path.join(model_dir, "comparative_3d_autoreg.gif"),
        sample_nodes=2000,
        frame_interval=1
    )
    print(f"Video comparativo guardado en: {comparative_gif}")

if __name__ == "__main__":
    main()