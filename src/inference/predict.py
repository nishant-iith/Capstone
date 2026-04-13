import torch
from src.models.gan import UNetGenerator
from PIL import Image
import numpy as np

def predict(model_path, image_path, output_path):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_t = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 127.5 - 1.0
    img_t = img_t.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_t)
        
    # Post-process output
    output = (output.squeeze(0).cpu().permute(1,2,0).numpy() + 1) / 2.0
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    
    Image.fromarray(output).save(output_path)

if __name__ == "__main__":
    print("Inference script ready.")
