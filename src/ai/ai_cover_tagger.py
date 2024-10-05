import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
from transformers import ResNetForImageClassification

# Mapping dictionary for numerical predictions to labels
LABEL_MAPPING = {0: "cover", 1: "other"}

def load_model(model_path):
    print(f"Loading complete model from {model_path}")
    model = torch.load(model_path)
    model.eval()
    print("Complete model loaded")
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def predict_batch(model, transform, image_paths, device):
    images = [transform(Image.open(image_path)).unsqueeze(0) for image_path in image_paths]
    images = torch.cat(images).to(device)

    with torch.no_grad():
        outputs = model(images).logits
    
    _, predicted = torch.max(outputs, 1)
    confidences = torch.nn.functional.softmax(outputs, dim=1) * 100

    predicted_labels = [LABEL_MAPPING[p.item()] for p in predicted]
    confidence_scores = [confidences[i][predicted[i]].item() for i in range(len(predicted))]

    return predicted_labels, confidence_scores

def process_images_in_folder(model, transform, folder_path, batch_size=32):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for subdir, _, files in os.walk(folder_path):
        true_label = os.path.basename(subdir)
        image_paths = [os.path.join(subdir, file) for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
        
        for i in tqdm(range(0, len(image_paths), batch_size), ascii=True, desc=f"Processing {true_label}"):
            batch_paths = image_paths[i:i + batch_size]
            predicted_labels, confidence_scores = predict_batch(model, transform, batch_paths, device)
            
            for j, image_path in enumerate(batch_paths):
                master_id = os.path.splitext(os.path.basename(image_path))[0]
                predicted_label = predicted_labels[j]
                confidence = confidence_scores[j]
                correct_guess = (predicted_label == true_label)
                results.append({
                    "master_id": master_id,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "correct_guess": correct_guess
                })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join('out', 'results 2.csv'), index=False)

def main():
    model_path = os.path.join('out', 'models', 'complete_model.pth')
    folder_path = os.path.join('out', 'images_sorted')

    model = load_model(model_path)
    transform = get_transform()
    process_images_in_folder(model, transform, folder_path)

if __name__ == "__main__":
    main()
