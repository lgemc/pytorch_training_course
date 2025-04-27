import matplotlib.pyplot as plt

def plot_losses(
        train_losses,
        test_losses,
        title:str,
        label1:str="Train Loss",
        label2:str="Test Loss",
):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label=label1, marker='o')
    plt.plot(epochs, test_losses, label=label2, marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
