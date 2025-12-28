# importing 3 files: 1 that has the fgsm method, 1 that has the noise gaussian attack method, and 1 that has the pretrained model setup
from mnist_model import model, test_loader
from fgsm import fgsm_attack
from fgsm_gaussian import fgsm_noise_attack


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Storing the accuracies of both attacks for further plotting
accuracies_noise = []
accuracies_fgsm = []


# Loop through test_loader
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)


    epsilon = 0.1

    # FGSM attack           
    _, _, acc_fgsm = fgsm_attack(model, nn.CrossEntropyLoss(), images, labels, epsilon)  # replace the first two underscores with a variable if u wnat those outputs which are : 1 - perturbed images 2 - perturbed outputs
    print(f"FGSM batch accuracy: {acc_fgsm*100:.2f}%")
    accuracies_fgsm.append(acc_fgsm)


    # Gaussian noise attack
    perturbed_images_noise, outputs_noise, acc_noise = fgsm_noise_attack(model, images, labels, epsilon)
    print(f"Gaussian Noise batch accuracy: {acc_noise*100:.2f}%")
    accuracies_noise.append(acc_noise)

    print('\n')



# Plot the accuracies
data = [accuracies_noise, accuracies_fgsm]
labels = ['Gaussian Noise', 'FGSM']

sns.boxplot(data=data)
plt.ylabel('Accuracy')
plt.xticks([0, 1], labels)  # Set x-axis labels
plt.title('Accuracy Distribution Comparison')
plt.show()

