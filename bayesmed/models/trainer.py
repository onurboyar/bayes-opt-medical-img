import torch
import torch.nn as nn

from bayesmed.utils.model_utils import *
from bayesmed.models.unet import UNet
from bayesmed.utils.discrete import *
from bayesmed.augmentations.list import get_augmentations
from bayesmed.utils.loaders import get_dataloaders
from bayesmed.utils.model_utils import dice_loss
from bayesmed.models.eval import evaluate

def train_unet(epochs=15, initial_lr = 0.001, scheduler = None, **params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr) #default lr is 0.001
    criterion = nn.CrossEntropyLoss()
 
    if "angle" in params.keys():
        params["angle"] = discrete_angle_normalized(params["angle"])

    if "shift_x" in params.keys():
        params["shift_x"] = discrete_shift(params["shift_x"])

    if "shift_y" in params.keys():
        params["shift_y"] = discrete_shift(params["shift_y"])

    transform = get_augmentations(**params)
    train_dataloader, test_dataloader = get_dataloaders(transform)
    best_dice = 0
    total = len(train_dataloader) * epochs

    with tqdm(total = total, desc='Training round', leave=False, position=0, ) as tt:
        for epoch in range(epochs):
            batch_count, train_loss = 0, 0
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch_count +=1
                optimizer.zero_grad()
                out = model(batch["image"].to(device))

                loss = criterion(out, batch["mask"].long().to(device)) \
                    + dice_loss(
                        F.softmax(out, dim=1).float(),
                        F.one_hot(batch["mask"].to(device), model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                )

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                tt.update()

            val_score = evaluate(model, test_dataloader, device, True)
            if val_score.item() > best_dice:
                best_dice = val_score.item()

            if scheduler != None:
                scheduler.step()

    return best_dice