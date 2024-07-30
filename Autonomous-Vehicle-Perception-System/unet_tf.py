import tensorflow as tf

class UNet(tf.keras.Model):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        
        # Downsampling path
        self.down1 = self.downsample(64, 4)
        self.down2 = self.downsample(128, 4)
        self.down3 = self.downsample(256, 4)
        self.down4 = self.downsample(512, 4)

        # Upsampling path
        self.up1 = self.upsample(512, 4)
        self.up2 = self.upsample(256, 4)
        self.up3 = self.upsample(128, 4)
        self.up4 = self.upsample(64, 4)

        # Final convolution
        self.final = tf.keras.layers.Conv2D(n_classes, 1)

    def downsample(self, filters, size):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, size, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters, size, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(2)
        ])

    def upsample(self, filters, size):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, size, 2, padding='same'),
            tf.keras.layers.Conv2D(filters, size, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters, size, padding='same', activation='relu')
        ])

    def call(self, x):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # Upsampling
        u1 = self.up1(d4)
        u2 = self.up2(tf.concat([u1, d3], axis=3))
        u3 = self.up3(tf.concat([u2, d2], axis=3))
        u4 = self.up4(tf.concat([u3, d1], axis=3))

        return self.final(u4)

# Training function
@tf.function
def train_step(images, masks, model, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(masks, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_unet(model, train_dataset, num_epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step, (images, masks) in enumerate(train_dataset):
            loss = train_step(images, masks, model, optimizer, loss_fn)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.numpy():.4f}")

# Main execution
if __name__ == "__main__":
    n_classes = 2  # Binary segmentation
    model = UNet(n_classes)
    
    # Load and preprocess your dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

    num_epochs = 100
    learning_rate = 0.001
    train_unet(model, train_dataset, num_epochs, learning_rate)
