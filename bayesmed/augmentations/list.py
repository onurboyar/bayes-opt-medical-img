from torchvision import transforms
from bayesmed.augmentations.functional as F

def get_augmentations(**params):
    transform = transforms.Compose([
        transforms.RandomApply([F.Rotate(params["angle"])], p=0.1), #0.1
        transforms.RandomApply([F.ShiftX(params["shift_x"])], p=0.2), #0.2
        transforms.RandomApply([F.ShiftY(params["shift_y"])], p=0.2), #0.2
        transforms.RandomApply([F.ZoomOut(params["zoom_amount"])], p=0.3), #0.3
        transforms.RandomApply([F.Gamma(params["gamma"])], p=0.3) #0.3
        #transforms.RandomApply([GaussianNoiseDeterministic(params["var"])], p=0.2)
    ])
    return transform
