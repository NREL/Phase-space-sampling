import os
import torch
from ..options import UIPSOptions

class DownSampler():
    def __init__(self, options: UIPSOptions):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        self.options = options

        self.configure_torch()
        self.configure_numpy()

    def configure_torch(self):
        """ Configure torch based on DownSamplerOptions
        """

        if self.options.use_gpu:
            device = torch.device("cuda")
            torch.set_default_dtype(torch.cuda.float32)
        else:
            device = torch.device("cpu")
            torch.set_default_dtype(torch.float32)
        torch.manual_seed(int(inpt["seed"]) + par.irank)

    def configure_numpy():
        numpy.random.seed(self.options.seed)

