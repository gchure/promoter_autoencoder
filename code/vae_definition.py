#%%
from fastai.vision.all import *

# Define the DataBlock and DataLoaders
def get_dls(path, bs=256):
    data = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), ImageBlock(cls=PILImageBW)),  # Both input and output are images
        get_items=get_image_files,        # Function to get all image files
        get_y=lambda x: x,                # Function to get the target (same as input)
        item_tfms=Resize((4, 40)))
    return data.dataloaders(path, bs=bs)

# Load the dataset
path = Path('../data/images/train/')
dls = get_dls(path)
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),  # (4, 40) -> (2, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),  # (2, 20) -> (1, 10)
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64 * 1 * 10, latent_dim)
        self.fc_logvar = nn.Linear(64 * 1 * 10, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 64 * 1 * 10)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 1)),  # (1, 10) -> (2, 20)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=(1, 1)),  # (2, 20) -> (4, 40)
            nn.Sigmoid()  # Output in range [0, 1]
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_decoded = self.fc_dec(z)
        x_decoded = x_decoded.view(-1, 64, 1, 10)
        x_recon = self.decoder(x_decoded)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



# Define the wrapped loss function
class VAELossWrapper:
    def __init__(self, vae_loss_func):
        self.vae_loss_func = vae_loss_func

    def __call__(self, pred, target):
        recon_x, mu, logvar = pred
        return self.vae_loss_func(recon_x, target, mu, logvar)




# Create an instance of the model
latent_dim = 20  
model = ConvVAE(latent_dim=latent_dim)

# Define the FastAI learner
learn = Learner(dls, model, loss_func=VAELossWrapper(vae_loss), metrics=[])

# Train the model
learn.fit_one_cycle(50, 1e-3)

#%%
def extract_latent_representations(learn, dl):
    latents, labels = [], []
    model = learn.model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        for xb, _ in dl:
            _, mu, logvar = model(xb)
            latents.append(mu.cpu().numpy())
            labels.append(_.cpu().numpy())
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    return latents, labels

latents, labels = extract_latent_representations(learn, dls.train)
plt.scatter(latents[:, 0], latents[:, 1])