import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import FOTSModel
from data_loaders.datasets import TotalText
from utils.data import collate_fn
from loss import DetectionLoss

def fit(model, data_loader, optimizer, criterion, n_epochs=2):
    for epoch in range(n_epochs):
        for batch_id, data in enumerate(data_loader):
            img_filenames, imgs, bboxes_true, texts, score_maps_true, geo_maps_true, angle_maps_true, training_masks, bbox_to_img_idx_true = data
            optimizer.zero_grad()
            score_maps_pred, geo_maps_pred, angle_maps_pred, bboxes_pred, bbox_to_img_idx_pred, seq_lens, logits = model(imgs, bboxes_true, bbox_to_img_idx_true)
            loss = criterion(score_maps_true, geo_maps_true, angle_maps_true, score_maps_pred, geo_maps_pred, angle_maps_pred, training_masks)
            loss.backward()
            optimizer.step()
            print(f"[{epoch+1}, {batch_id+1}] loss: {loss:.4f}")

if __name__ == "__main__":
    model = FOTSModel(is_training=True)
    total_text = TotalText('')
    train_loader = DataLoader(total_text, batch_size=2, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = DetectionLoss()
    fit(model, train_loader, optimizer, criterion)