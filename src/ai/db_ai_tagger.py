import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import ResNetForImageClassification
import pymongo
from io import BytesIO
from dotenv import load_dotenv

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
    images = [transform(Image.open(BytesIO(image_data))).unsqueeze(0) for image_data in image_paths]
    images = torch.cat(images).to(device)

    with torch.no_grad():
        outputs = model(images).logits
    
    _, predicted = torch.max(outputs, 1)
    confidences = torch.nn.functional.softmax(outputs, dim=1) * 100

    predicted_labels = [LABEL_MAPPING[p.item()] for p in predicted]
    confidence_scores = [confidences[i][predicted[i]].item() for i in range(len(predicted))]

    return predicted_labels, confidence_scores

def update_documents_bulk(collection, updates):
    bulk_operations = []
    for update in updates:
        master_id = update["master_id"]
        predicted_label = update["predicted_label"]
        confidence_score = update["confidence_score"]
        print(f"Preparing update for master_id: {master_id} with ai_tag: {predicted_label} and ai_confidence: {confidence_score}")
        bulk_operations.append(
            pymongo.UpdateOne(
                {"master_id": master_id},
                {"$set": {"ai_tag": predicted_label, "ai_confidence": confidence_score}}
            )
        )
    
    if bulk_operations:
        result = collection.bulk_write(bulk_operations)
        print(f"Matched {result.matched_count} documents and modified {result.modified_count} documents.")

def process_batch(model, transform, documents, device):
    print("Processing batch of images")
    image_data_list = [doc["image_data"] for doc in documents]
    master_ids = [doc["master_id"] for doc in documents]

    predicted_labels, confidence_scores = predict_batch(model, transform, image_data_list, device)

    updates = []
    for master_id, predicted_label, confidence_score in zip(master_ids, predicted_labels, confidence_scores):
        updates.append({
            "master_id": master_id,
            "predicted_label": predicted_label,
            "confidence_score": confidence_score
        })

    print(updates)
    return updates

def process_images_in_db(model, transform, collection, sort_field='master_id', batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_documents = collection.count_documents({"image_data": {"$exists": True}, "ai_tag": None})
    print(f"Total documents to process: {total_documents}")
    for i in tqdm(range(0, total_documents, batch_size), ascii=True, desc="Processing images"):
        documents = list(collection.find(
            {"image_data": {"$exists": True}, "ai_tag": None}
        ).sort(sort_field, 1).limit(batch_size).allow_disk_use(True))
        
        updates = process_batch(model, transform, documents, device)
        update_documents_bulk(collection, updates)

def connect_to_db():
    load_dotenv()
    mongo_uri = os.getenv("MONGODB_URI")
    client = pymongo.MongoClient(mongo_uri)
    db = client["album_covers"]
    collection = db["fiveK-albums-sample-copy"]
    return collection

def main():
    collection = connect_to_db()

    model_path = os.path.join('out', 'models', 'complete_model.pth')
    model = load_model(model_path)
    transform = get_transform()
    batch_size = 16

    process_images_in_db(model, transform, collection, batch_size=batch_size)

if __name__ == "__main__":
    main()
