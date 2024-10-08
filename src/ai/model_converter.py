import torch
from transformers import ResNetForImageClassification

def save_complete_model(model_path, save_path):
    print(f"Loading model from {model_path}")
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50", 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("Model loaded")

    # Save the complete model
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    model_path = 'out/models/best_model.pth'
    save_path = 'out/models/complete_model.pth'
    save_complete_model(model_path, save_path)
