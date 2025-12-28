import torch

def fgsm_noise_attack(model,images, labels, epsilon = 0.1):
  was_training = model.training
  device = next(model.parameters()).device
  images, labels = images.clone().detach().to(device), labels.to(device)

  # putting the model in evaluation mode....this is not a necessesity for noise based fgsm but it still saves us from unprecedented bugs
  model.eval()

  # making a new tensor with same shape of the image tensors..we will add this tensor to the image to perturb it
  noise = torch.randn_like(images) * epsilon

  # Create adversarial example..adding noise in the pixels
  perturbed_images = images + noise
  # clamping to keep pixel values valid between 0 and 1
  perturbed_images = torch.clamp(perturbed_images, 0, 1)

  # getting the predictions on the perturbed data-images
  perturbed_outputs = model(perturbed_images)

  #calculating accuracy on the perturbated data
  _, predicted = torch.max(perturbed_outputs, 1)
  correct = (predicted == labels).sum().item()
  accuracy = correct / labels.size(0)

  # going back to training mode...i did it here in case in the future when training i forget to turn this mode on
  if was_training:
        model.train()

  return perturbed_images, perturbed_outputs, accuracy    # returning all 3 for more information

