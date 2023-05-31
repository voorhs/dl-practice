from PIL import ImageEnhance
from glob import glob
from PIL import Image
import numpy as np

import torchvision.transforms as T
from torch.utils.data import Dataset


class PhotosDataset(Dataset):
    def __init__(self, images_dir, target_dir=None, transforms=None):
        """
        Arguments
        ---------
        images_dir : str
            Path to directory with images

        target_dir : str
            Path to directory with masks.
            Each mask corresponds to one image.
            Corresponding mask and image have the same name, but different format.

        transforms : some collection
            Sequence of transformations for images and masks. 
        """
        self.images_dir = images_dir
        self.target_dir = target_dir

        self.images_file_names = sorted(glob(images_dir + '/*.jpg'))
        self.target_file_names = None
        if target_dir is not None:
            self.target_file_names = sorted(glob(target_dir + '/*.png'))

        self.transforms = transforms

    def __len__(self):
        return len(self.images_file_names)

    def __getitem__(self, idx):
        """
        Arguments
        ---------
        idx : int
            Index of image and mask

        Returns
        -------
        (image, mask)
        """

        image = Image.open(self.images_file_names[idx])
        mask = Image.open(self.target_file_names[idx])

        n = len(self.transforms)
        events = np.random.uniform(low=0, high=1, size=n)
        for i, trans in enumerate(self.transforms):
            if events[i] < trans.p:
                image = trans(image, mask)
                if trans.mask_too:
                    if isinstance(trans, RandomCrop):
                        mask = mask.crop(trans.box).resize(mask.size)
                    else:
                        mask = trans(mask)

        return image, mask


class HorizontalFlip(object):
    mask_too = True

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask=None):
        """
        Args:
            img (PIL.Image): image to be flipped.

        Returns:
            PIL.Image: flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)


class RandomCrop(object):
    mask_too = True

    def __init__(self, size, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, img, mask=None):
        """
        Args:
            img (PIL.Image): image to be cropped.

        Returns:
            PIL.Image: cropped image.
        """

        x1 = np.random.randint(low=0, high=img.size[0]-self.size[0])
        y1 = np.random.randint(low=0, high=img.size[1]-self.size[1])
        self.box = (x1, y1, x1+self.size[0], y1+self.size[1])
        return img.crop(self.box).resize(img.size)


class AdjustBrightness(object):
    mask_too = False

    def __init__(self, factor=None, p=0.5):
        self.factor = factor
        self.p = p

    def __call__(self, img, mask=None):
        """
        Args:
            img (PIL.Image): image to be brightened of darkened.

        Returns:
            PIL.Image: brightened or darkened image.
        """

        self.enh = ImageEnhance.Brightness(img)

        if self.factor is None:
            return self.enh.enhance(np.random.uniform(low=0.75, high=1.25))
        return self.enh.enhance(self.factor)


class RandomBackground(object):
    mask_too = False

    def __init__(self, backs_dir='backs', p=0):
        self.p = p
        back_file_names = glob(backs_dir + '/*.png')
        self.backs = []
        for filename in back_file_names:
            self.backs.append(Image.open(filename))

    def __call__(self, img, mask):
        """
        Args:
            img (PIL.Image): image to be flipped
            mask (PIL.Image): 0/1 mask of object on image

        Returns:
            PIL.Image: image with randomly changed background.
        """

        back_ind = np.random.randint(low=0, high=len(self.backs))

        return Image.composite(img, self.backs[back_ind], mask)


class ToTensor(T.ToTensor):
    p = 1
    mask_too = True

    def __call__(self, img, mask=None):
        return T.functional.to_tensor(pic=img)


class Normalize(T.Normalize):
    p = 1
    mask_too = False

    def __call__(self, img, mask=None):
        return super(Normalize, self).__call__(img)
