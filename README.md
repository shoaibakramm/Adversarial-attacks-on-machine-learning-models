# Adversarial-attacks-on-machine-learning-models

In this repo i replicate adversarial attacks and test them on machine learning models. I use the FGSM(fast gradient sign method) and the gaussian noise method for the pertubation of the inputs to achieve the goal. Additionally, i have also implemented the prompt injection adversarial attack on large language models using the Universal prompt injection.


Gaussian (random) noise attack.(black box attack)
FGSM (gradient-based) attack.(white box attack)


fgsm.py
    • Implements the Fast Gradient Sign Method to generate adversarial examples. This tests the white box method of attack.
    • Returns perturbed images, model predictions, and attack success/accuracy.

fgsm_gaussian.py
    • Replacing the gradient perturbation with Gaussian noise.
    • Evaluates robustness of models under noisy adversarial conditions. This tests the black box method of attack.
    - Returns perturbed images, model predictions, and attack success/accuracy

attack_file.py
    • Contains batched examples going through both fgsm and gaussian attack.
    • Calculates accuracies for both and also plots a box plot

mnist_model.py
    • Has the pretrained model setup
    • initializes the model

app_fgsm.py
    • Wraps fgsm.py into a RESTful API using FastAPI.
    • Accepts an input image, label, and epsilon, and returns success status and predictions.

mnist_model.pth
  - I have included the pretrained weights of the model in this zip file. This zip is being loaded in hte mnist_model.py file.


![Accuracies](Accuracies.png)

![Plotted Accuracies](Accuracies_plotted.png)








