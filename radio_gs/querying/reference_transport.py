"""Lazy reference/target RGB preparation shared by signed-reference transport."""
from PIL import Image
from torchvision.transforms.v2 import functional as TF


class LazyImages:
    def __init__(self, paths, wh, transform):
        self.paths, self.wh, self.transform = paths, wh, transform

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB').resize(self.wh, Image.Resampling.BILINEAR)
        return self.transform(TF.to_image(image))
