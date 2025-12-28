from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

import torch
from torchvision import transforms

from PIL import Image
import io

from fgsm import fgsm_attack
from mnist_model import model

app = FastAPI(title="FGSM Attack API")




loss_fn = torch.nn.CrossEntropyLoss()

# Transform incoming image to tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale for MNIST
    transforms.Resize((28, 28)),                  # MNIST size
    transforms.ToTensor()
])



@app.post("/fgsm_attack/")
async def run_fgsm_attack(
    file: UploadFile = File(...),
    epsilon: float = Form(...),
    label: int = Form(...)
):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        label_tensor = torch.tensor([label])

        # Run FGSM attack
        perturbed_images, perturbed_outputs, accuracy = fgsm_attack(
            model=model,
            loss_type=loss_fn,
            images=image_tensor,
            labels=label_tensor,
            epsilon=epsilon
        )

        success = accuracy < 1.0  # True if attack caused misclassification

        return JSONResponse(
            {
            "success": success,
            "accuracy": accuracy,
            "predicted_label": int(torch.argmax(perturbed_outputs, 1).item())
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)