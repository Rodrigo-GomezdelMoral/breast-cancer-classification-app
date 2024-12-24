import torch
import torchvision.transforms as transforms
from PIL import Image

def load_model():
    """
    Load the DL-model
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load("utils/best_model.h5", map_location=device)
    model.eval()

    return device, model

def predict_image(image_path, device, model):
    """
    Process the image and predict its class.
    """

    class_names = ['Benign', 'Malignant']

    transform = transforms.Compose([
        transforms.Resize((460, 460)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    return class_names[predicted_class.item()]