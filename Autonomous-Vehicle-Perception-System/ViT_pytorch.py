import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(ViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads),
            num_layers=depth
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img):
        # Extract patches
        p = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        p = p.contiguous().view(img.size(0), -1, 3 * self.patch_size ** 2)
        
        # Patch embedding
        x = self.patch_embedding(p)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x += self.pos_embedding
        
        # Apply transformer
        x = self.transformer(x)
        
        # MLP head
        x = x[:, 0]
        return self.mlp_head(x)

# Training function
def train_vit(model, train_loader, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

# Main execution
if __name__ == "__main__":
    image_size = 224
    patch_size = 16
    num_classes = 1000
    dim = 768
    depth = 12
    heads = 12
    mlp_dim = 3072
    
    model = ViT(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)
    
    # Load and preprocess your dataset
    train_dataset = YourDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_epochs = 100
    learning_rate = 0.001
    train_vit(model, train_loader, num_epochs, learning_rate)
