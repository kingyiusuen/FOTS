import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import FOTSModel
from data_loaders.datasets import TotalText
import modules.alphabet
from utils.data import collate_fn
from utils.tokenizer import Tokenizer
from loss import FOTSLoss

def fit(model, data_loader, optimizer, criterion, tokenizer, n_epochs=2):
    for epoch in range(n_epochs):
        for batch_id, data in enumerate(data_loader):
            # get ground truth labels
            img_filenames, imgs, bboxes_true, texts, score_maps_true, geo_maps_true, angle_maps_true, training_masks, bbox_to_img_idx_true = data
            indexed_tokens_true, seq_lens_true = tokenizer.encode(texts)
            # zero the gradients
            optimizer.zero_grad()
            # get predictions
            score_maps_pred, geo_maps_pred, angle_maps_pred, bboxes_pred, bbox_to_img_idx_pred, log_probs, seq_lens_pred = model(imgs, bboxes_true, bbox_to_img_idx_true)
            # compute the looss
            detect_loss, recog_loss, loss = criterion(
                score_maps_true, geo_maps_true, angle_maps_true, 
                score_maps_pred, geo_maps_pred, angle_maps_pred, training_masks,
                log_probs, indexed_tokens_true, seq_lens_pred, seq_lens_true
            )
            # backprop and update weights
            loss.backward()
            optimizer.step()
            print(f"[{epoch+1}, {batch_id+1}] detection loss: {detect_loss:.4f}, recognition loss: {recog_loss:.4f}, total loss: {loss:.4f}")

if __name__ == "__main__":
    model = FOTSModel(is_training=True)
    total_text = TotalText('')
    tokenizer = Tokenizer(modules.alphabet.CHARS)
    train_loader = DataLoader(total_text, batch_size=2, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = FOTSLoss()
    fit(model, train_loader, optimizer, criterion, tokenizer)