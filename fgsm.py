import torch

def fgsm_attack(model, loss_type, images, labels, epsilon):
  device = next(model.parameters()).device
  images, labels = images.clone().detach().to(device), labels.to(device)

  images.requires_grad = True

  # putting the model in evaluation mode
  model.eval()


  #Forward pass
  outputs = model(images)
  loss = loss_type(outputs, labels)

  #Backward pass
  if images.grad is not None:
        images.grad.zero_()

  loss.backward()

  #Collect gradient of the input and then storing the sign of it
  grad_sign = images.grad.sign()

  # Create adversarial example..moving the pixels in the direction on the loss increase
  perturbed_images = images + epsilon * grad_sign
  # clamping to keep pixel values valid between 0 and 1
  perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()

  # getting the predictions on the perturbed data-images
  perturbed_outputs = model(perturbed_images)

  #calculating accuracy on the perturbated data
  _, predicted = torch.max(perturbed_outputs, 1)
  correct = (predicted == labels).sum().item()
  accuracy = correct / labels.size(0)

  # going back to training mode...i did it here in case in the future when training i forget to turn this mode on
  model.train()

  return perturbed_images, perturbed_outputs, accuracy    # returning all 3 for more information

