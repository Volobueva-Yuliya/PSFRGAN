from dcpsfrgan.data.base_dataset import BaseDataset, get_transform
from dcpsfrgan.data.image_folder import make_dataset
from PIL import Image
import numpy as np

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.src_dir, opt.max_dataset_size))
        input_nc = self.opt.output_nc 
        self.transform = get_transform(opt, grayscale=(input_nc == 1))
        self.opt = opt
 
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A_img = A_img.resize((512, 512), Image.BICUBIC)

        A = self.transform(A_img)
        return {'LR': A, 'LR_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
