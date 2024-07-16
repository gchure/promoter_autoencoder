#%%
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import os 
from torch.utils.data import Dataset, DataLoader
import tqdm
import torchvision.transforms as transforms
import numpy as np 
import pandas as pd 
import glob
from PIL import Image
import matplotlib.pyplot as plt
files = glob.glob('../data/images/train/*.png')
class VAE(nn.Module):
    """
    Base class for a variational autoencoder for 4x40 images of one-hot 
    encoded DNA sequences.
    """
    def __init__(self, latent_dim=10):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Define the encoder architecture as a convolutional neural network.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Set the latent space parameters
        self.fc_mu = nn.Linear(320, latent_dim)
        self.fc_logvar = nn.Linear(320, latent_dim)

        # Define the decoder architecture 
        self.fc_decode = nn.Linear(latent_dim, 320)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32,1, 10)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1,1)),
            nn.Sigmoid() # for the binary output
        )
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for the VAE.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        enc = self.encoder(x)
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        return mu, logvar
    
    def decode(self, x):
        dec = self.fc_decode(x)
        return self.decoder(dec)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Define a custom Dataset class for the images
class PromoterImages(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((4, 40)),  # Resize images to 4x40
            transforms.ToTensor()  # Convert to tensor
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, img_name #float(img_name.split('/')[-1].split('-')[0])

# With the VAE defined, define the loss function
def loss_function(output, input, mu, logvar):
    binary_cross_ent = F.binary_cross_entropy(output, input, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return binary_cross_ent + kl_div

#%% Train
# Set the training image loader
dataset = PromoterImages('../data/images/train/')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#%%
# Instantiate the  vae
vae = VAE(latent_dim=10)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
# Training loop
epochs = 500 
loss_step = np.zeros(epochs)
for epoch in tqdm.tqdm(range(epochs)):
    vae.train()
    train_loss = 0
    for batch_idx, (data, labs) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    loss_step[epoch] = train_loss
    # print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')
#%%
import matplotlib.pyplot as plt
plt.plot(loss_step)
#%%
# Examine the latent space embedding.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae.to(device)
def extract_latent_variables(vae, data_loader):
    vae.eval()
    df = pd.DataFrame([])
    with torch.no_grad():
        for data, name in data_loader:
            data = data.to(device)
            mu, _ = vae.encode(data)
            _df = pd.DataFrame(mu, columns=[f'dim_{i+1}' for i in range(mu.shape[1])])
            _names = np.array([n.split('/')[-1].split('-')[0] for n in name])
            _df['promoter_id'] = _names.astype(int)
            df = pd.concat([df, _df])
    return df
df = extract_latent_variables(vae, train_loader)

#%%
from utils import matplotlib_style
cor, pal = matplotlib_style()
seqs = pd.read_csv('../data/promoter_sequences.csv')
seqs['promoter_id'] = seqs['promoter_id'].astype(int)
merged = seqs.merge(df, on='promoter_id')
_merged = merged[(merged['sigma_factors'].isin(['RpoS', 'RpoN', 'RpoH', 'FliA', 'RpoE', 'FecI']))]
mapper = {'RpoS':cor['primary_green'],
          'RpoD':cor['primary_black'],
          'RpoN':cor['primary_blue'],
          'RpoH':cor['primary_blue'],
          'FliA':cor['primary_red'],
          'RpoE':cor['primary_blue'],
          'FecI':cor['primary_gold']}
_merged['color'] = _merged['sigma_factors'].map(mapper) 
_merged
#%%
dims = 10
fig, ax = plt.subplots(dims, dims, figsize=(10, 10))
for a in ax.ravel():
    a.set_xticks([])
    a.set_yticks([])
for i in range(dims):
    for j in range(dims):
        # if (i >= j):
        ax[i, j].scatter(_merged[f'dim_{i+1}'], _merged[f'dim_{j+1}'], marker='o', 
                             s=4, c=_merged['color'], alpha=0.5) 

        # else:
            # ax[i,j].axis(False)


# Dplt.plot(df['dim_1'], df['dim_4'], '+')

