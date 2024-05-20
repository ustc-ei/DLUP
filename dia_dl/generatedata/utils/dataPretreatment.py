import numpy as np
import numpy.typing as npt


def ionMobilityPretreatment(
    ionmobility_sequence: npt.NDArray,
    massspectrum_peaknum: int
):
    """
    拓展为向量
    [0.1, 0.2, 0.3] > [[0.1, 0.1, 0.1],
                       [0.2, 0.2, 0.2],
                       [0.3, 0.3, 0.3]]
    """
    result = []
    for ion_mobility in ionmobility_sequence:
        ion_mobility_extension = np.repeat(
            ion_mobility, massspectrum_peaknum)
        result.append(ion_mobility_extension)

    return np.array(result)


def featuredIonsPretreatment(
    featured_ions: npt.NDArray
):
    return featured_ions.reshape(featured_ions.shape[0], featured_ions.shape[1], 1)
