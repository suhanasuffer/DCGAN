import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import kagglehub
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download LSUN Bedrooms dataset
path = kagglehub.dataset_download("jhoward/lsun_bedroom")
print("Path to dataset files:", path)

data_root = os.path.join(path, "lsun_bedroom")

# Hyperparameters
image_size = 64
batch_size = 128
nz = 100  
lr = 0.0002
beta1 = 0.5
num_epochs = 50

# Transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load dataset
dataset = ImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define Generator class
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Define Discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

# Initialize models
generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Update Discriminator
        discriminator.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels_real = torch.ones(batch_size, 1, device=device)
        labels_fake = torch.zeros(batch_size, 1, device=device)
        
        output_real = discriminator(real_images).view(-1, 1)
        loss_real = criterion(output_real, labels_real)
        
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach()).view(-1, 1)
        loss_fake = criterion(output_fake, labels_fake)
        
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()
        
        # Update Generator
        generator.zero_grad()
        output_fake = discriminator(fake_images).view(-1, 1)
        loss_g = criterion(output_fake, labels_real)
        loss_g.backward()
        optimizer_g.step()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")
    
    # Save sample images
    if epoch % 5 == 0:
        vutils.save_image(fake_images, f"generated_epoch_{epoch}.png", normalize=True)

# Save trained model
torch.save(generator.state_dict(), "dcgan_generator.pth")
torch.save(discriminator.state_dict(), "dcgan_discriminator.pth")

print("Training complete! Models saved.")
