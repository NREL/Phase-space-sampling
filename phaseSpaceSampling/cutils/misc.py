from phaseSpaceSampling.utils.torchutils import tensor2numpy


def gridimshow(image, ax):
    if image.shape[0] == 1:
        image = tensor2numpy(image[0, ...])
        ax.imshow(1 - image, cmap="Greys")
    else:
        image = tensor2numpy(image.permute(1, 2, 0))
        ax.imshow(image)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_xticklabels("")
    ax.set_yticklabels("")