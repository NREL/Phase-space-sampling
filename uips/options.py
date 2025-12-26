from dataclasses import dataclass, field, asdict
import os
import typing as t
import enum

class PDFMethodEnum(enum.Enum):
    NORMALIZINGFLOW = enum.auto()
    BINS = enum.auto()

    def lookup(self, name):
        for member in self:
            if member.name.lower() == name.lower():
                return member
            raise ValueError(f"PDFMethodEnum has no method {name}")

@dataclass
class stepOptions:
    # Number of epochs for each normalizing flow training
    nEpochs: int = 5
    # Batch size for normalizing flow training
    batch_size_train: int = 25_000
    # Subset of data on which training is done
    nWorkingData: int = 10_000
    data_freq_adjustment: int = 1
    # Learning rate during normalizing flow training
    learning_rate: float = 2e-4
    num_bins: int = 4
    hidden_features: int = 12
    nCouplingLayer: int = 2
    num_blocks: int = 3

@dataclass
class UIPSOptions:
    num_pdf_iter: int = 2
    stepOptionsList: t.List[stepOptions] = field(default_factory = lambda : [stepOptions(), stepOptions()])
    pdfMethod: str = "NormalizingFlow"
    batch_size_eval: int = 2.5e4
    seed: int = 42
    printTiming: bool = False
    nDatReduced: int = -1
    nDimReduced: int = -1
    dimList: t.Optional[int] = None
    dataFile: "os.PathLike[str]" = "./datafile.npy"
    preShuffled: bool = True
    scalerFile: "os.PathLike[str]" = "scaler.npz"
    nSamples_list: t.List[int] = field(default_factory=lambda : [1e3, 1e4, 1e5])
    nSampleCriterionLimit: int = 1e5
    computeDistanceCriterion: bool = True
    prefixDownsampledData: str = "downSampledData"
    use_gpu: bool = True
    num_pdf_bins: int = 20
    data_freq_adjustment: int = 1
    # Subset of data used to adjust the sampling probability to the desired number of samples
    # If nWorkingDataAdjustment < 0, all data with the prescribed frequency is used
    nWorkingDataAdjustment: int = -1

    @property
    def nSamples(self) -> t.List[int]:
        return [int(n) for n in self.nSamples_list]

    def __post_init__(self):
        if len(self.stepOptionsList) != self.num_pdf_iter:
            raise Exception("Length of stepOptions is less than the number of pdf iterations")

    def stepOptions_as_list(self, field):
        return [asdict(step_option)[field] for step_option in self.stepOptionsList]