import argparse
import scanpy as sc
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scvi

# ==========================================
# 1. Custom Deep Learning Models (PyTorch)
# ==========================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc21 = nn.Linear(256, latent_dim) # Mean
        self.fc22 = nn.Linear(256, latent_dim) # Log Variance
        
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function_vae(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# ==========================================
# 2. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Latent Representation Learning for scRNA-seq")
    parser.add_argument("--input", required=True, help="Path to preprocessed .h5ad file")
    parser.add_argument("--output", required=True, help="Path to save .h5ad with latent representations")
    parser.add_argument("--method", choices=["pca", "ae", "vae", "scvi"], required=True, help="Latent model to use")
    parser.add_argument("--dim", type=int, default=50, help="Latent dimensionality")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for DL models")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    print(f"[*] Loading preprocessed data from {args.input}...")
    adata = sc.read_h5ad(args.input)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # --- PCA ---
    if args.method == "pca":
        print(f"[*] Running PCA ({args.dim} dimensions)...")
        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()
        X = X - X.mean(axis=0)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        sign_flip = np.sum(Vt, axis=1) < 0
        U[:, sign_flip] *= -1
        X_pca = U @ np.diag(s)
        order = np.argsort(s)[::-1]
        X_pca = X_pca[:, order][:, :args.dim]
        adata.obsm["X_pca"] = X_pca.copy()
        adata.obsm[f"X_latent_{args.method}"] = X_pca.copy()

    # --- scVI ---
    elif args.method == "scvi":
        print(f"[*] Running scVI ({args.dim} dimensions)...")
        # scVI uses raw counts from layered data
        scvi.model.SCVI.setup_anndata(adata, layer="counts")
        model = scvi.model.SCVI(adata, n_latent=args.dim, gene_likelihood="zinb")
        # Use simple progress bar for cleaner logs
        model.train(max_epochs=args.epochs, batch_size=args.batch_size, plan_kwargs={"n_epochs_kl_warmup": min(400, args.epochs)})
        adata.obsm[f"X_latent_{args.method}"] = model.get_latent_representation()

    # --- Autoencoder & VAE ---
    elif args.method in ["ae", "vae"]:
        print(f"[*] Running {args.method.upper()} ({args.dim} dimensions)...")
        input_dim = adata.n_vars
        
        # Prepare tensor data (using log-normalized data from adata.X)
        dense_X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        tensor_X = torch.tensor(dense_X, dtype=torch.float32)
        dataset = TensorDataset(tensor_X)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        if args.method == "ae":
            model = Autoencoder(input_dim=input_dim, latent_dim=args.dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            for epoch in range(args.epochs):
                model.train()
                epoch_loss = 0
                for batch in dataloader:
                    batch_x = batch[0].to(device)
                    optimizer.zero_grad()
                    recon_x, _ = model(batch_x)
                    loss = criterion(recon_x, batch_x)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss / len(dataset):.4f}")

            model.eval()
            with torch.no_grad():
                _, latent_space = model(tensor_X.to(device))
                adata.obsm[f"X_latent_{args.method}"] = latent_space.cpu().numpy()

        elif args.method == "vae":
            model = VAE(input_dim=input_dim, latent_dim=args.dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(args.epochs):
                model.train()
                epoch_loss = 0
                for batch in dataloader:
                    batch_x = batch[0].to(device)
                    optimizer.zero_grad()
                    recon_x, mu, logvar = model(batch_x)
                    loss = loss_function_vae(recon_x, batch_x, mu, logvar)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs}, Loss (ELBO): {epoch_loss / len(dataset):.4f}")

            model.eval()
            with torch.no_grad():
                mu, _ = model.encode(tensor_X.to(device))
                adata.obsm[f"X_latent_{args.method}"] = mu.cpu().numpy()

    # Save
    print(f"[*] Saving AnnData with {args.method} latent space to {args.output}...")
    adata.write(args.output)
    print("[*] Done!")

if __name__ == "__main__":
    main()
