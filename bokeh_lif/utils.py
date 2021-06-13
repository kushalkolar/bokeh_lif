from typing import *
import pathlib
import javabridge
import bioformats
import xmltodict
from functools import lru_cache
from explore_lif import Reader


# this needs to be here for bioformats
javabridge.start_vm(class_path=bioformats.JARS)


def get_series_indices(lif_reader: Reader) -> Dict[str, int]:
    return {s.getName().lower(): i for i, s in enumerate(lif_reader.getSeries())}


@lru_cache(maxsize=32)
def _get_series_mapping_list(lif_path: Union[pathlib.Path, str]):
    ome = bioformats.get_omexml_metadata(
        lif_path.as_posix() if isinstance(lif_path, pathlib.Path) else lif_path
    )
    d = xmltodict.parse(ome)

    series_mapping_list = []
    for i in range(len(d['OME']['StructuredAnnotations']['XMLAnnotation'])):
        if 'FilterSettingRecord|Laser wavelength #' in \
                d['OME']['StructuredAnnotations']['XMLAnnotation'][i]['Value']['OriginalMetadata']['Key']:
            series_mapping_list.append(
                d['OME']['StructuredAnnotations']['XMLAnnotation'][i]['Value']['OriginalMetadata']
            )

    return series_mapping_list


def get_channel_mapping(lif_path: Union[pathlib.Path, str], series: str) -> Dict[int, int]:
    """
    Returns dict with format: {channel_ix: laser_wavelength}

    Parameters
    ----------
    lif_path: Union[pathlib.Path, str]
        Path to the lif file
    series: str
        Name of the series
    Returns
    -------
    """

    series_mapping_list = _get_series_mapping_list(lif_path)

    series = series.lower()

    out = {}
    for s in series_mapping_list:
        if s['Key'].lower().startswith(series):
            k = s['Key'].lower()
            if not k[-2] == '#':
                raise ValueError(f"Unexpected formatting for laser wavelength.\n"
                                 f"Check key: {k}\n")
            channel_ix = int(k[-1]) - 1
            laser_id = int(s['Value'])

            if channel_ix in out.keys():
                raise KeyError(f"Duplicate channel # found\n"
                               f"Check key: {k}")
            if laser_id in out.values():
                raise ValueError(f"Duplicate laser wavelength found\n"
                                 f"Check key: {k}")
            out[laser_id] = channel_ix

    return {v: k for k, v in out.items()}  # invert
