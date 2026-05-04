# Output from running train.py (taken from my console).
## Epoch  1/3  Train Loss: 0.3445  Val Loss: 0.3077
## Epoch  2/3  Train Loss: 0.2226  Val Loss: 0.3116
## Epoch  3/3  Train Loss: 0.1698  Val Loss: 0.1518

import matplotlib.pyplot as plt

train_loss = [0.3445, 0.2226, 0.1698]
val_loss = [0.3077, 0.3116, 0.1518]
epochs = [1,2,3]

plt.plot(epochs, train_loss, color='purple', label='Train Loss')
plt.plot(epochs, val_loss, color='orange', label='Val Loss')
plt.legend()
plt.savefig("Output/loss_graph.png")
plt.show()