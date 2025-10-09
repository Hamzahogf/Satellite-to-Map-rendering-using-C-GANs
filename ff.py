import json
import matplotlib.pyplot as plt 

# Load file
with open("./stats.json", "r") as f:
    data = json.load(f)

# Extract losses
train_G = data["train_loss"]["G"]
train_D = data["train_loss"]["D"]

# Total loss = G + D
train_total = [g + d for g, d in zip(train_G, train_D)]

# Plot
plt.figure(figsize=(10,6))
plt.plot(train_G, label="Generator Loss (G)")
plt.plot(train_D, label="Discriminator Loss (D)")
plt.plot(train_total, label="Total Loss (G+D)", linestyle="--")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()
