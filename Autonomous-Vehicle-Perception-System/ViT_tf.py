import tensorflow as tf

class ViT(tf.keras.Model):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(ViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size
        self.patch_embedding = tf.keras.layers.Dense(dim)
        self.pos_embedding = self.add_weight("pos_embedding", shape=(1, num_patches + 1, dim))
        self.cls_token = self.add_weight("cls_token", shape=(1, 1, dim))
        self.transformer = tf.keras.Sequential([
            tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim) for _ in range(depth)
        ])
        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(mlp_dim, activation='gelu'),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, img):
        batch_size = tf.shape(img)[0]
        
        # Extract patches
        patches = tf.image.extract_patches(
            images=img,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [batch_size, -1, 3 * self.patch_size ** 2])
        
        # Patch embedding
        x = self.patch_embedding(patches)
        
        # Add classification token
        cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        x = tf.concat([cls_tokens, x], axis=1)
        
        # Add position embedding
        x = x + self.pos_embedding
        
        # Apply transformer
        x = self.transformer(x)
        
        # MLP head
        x = x[:, 0]
        return self.mlp_head(x)

# Training function
@tf.function
def train_step(images, labels, model, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_vit(model, train_dataset, num_epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step, (images, labels) in enumerate(train_dataset):
            loss = train_step(images, labels, model, optimizer, loss_fn)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.numpy():.4f}")

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
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

    num_epochs = 100
    learning_rate = 0.001
    train_vit(model, train_dataset, num_epochs, learning_rate)
