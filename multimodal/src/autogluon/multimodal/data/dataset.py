import logging
import torch
import numpy as np
import pandas as pd
from typing import List

from ..constants import AUTOMM, GET_ITEM_ERROR_RETRY
from .preprocess_dataframe import MultiModalFeaturePreprocessor

logger = logging.getLogger(AUTOMM)


class BaseDataset(torch.utils.data.Dataset):
    """
    A Pytorch DataSet class to process a multimodal pd.DataFrame. It first uses a preprocessor to
    produce model-agnostic features. Then, each processor prepares customized data for one modality
    per model. For code simplicity, here we treat ground-truth label as one modality. This class is
    independent of specific data modalities and models.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: List[MultiModalFeaturePreprocessor],
        processors: List[dict],
        is_training: bool = False,
    ):
        """
        Parameters
        ----------
        data
            A pd.DataFrame containing multimodal features.
        preprocessor
            A list of multimodal feature preprocessors generating model-agnostic features.
        processors
            Data processors customizing data for each modality per model.
        is_training
            Whether in training mode. Some data processing may be different between training
            and validation/testing/prediction, e.g., image data augmentation is used only in
            training.
        """
        super().__init__()
        self.processors = processors
        self.is_training = is_training
        self._consecutive_errors = 0

        self.lengths = []
        for i, (per_preprocessor, per_processors_group) in enumerate(zip(preprocessor, processors)):
            for per_modality in per_processors_group:
                per_modality_features = getattr(per_preprocessor, f"transform_{per_modality}")(data)
                setattr(self, f"{per_modality}_{i}", per_modality_features)
                self.lengths.append(len(per_modality_features[next(iter(per_modality_features))]))
        assert len(set(self.lengths)) == 1

    def __len__(self):
        """
        Assume that all modalities have the same sample number.

        Returns
        -------
        Sample number in this dataset.
        """
        return self.lengths[0]

    def __getitem__(self, idx):
        """
        Iterate through all data processors to prepare model inputs. The data processors are
        organized first by modalities and then by models.

        Parameters
        ----------
        idx
            Index of sample to process.

        Returns
        -------
        Input data formatted as a dictionary.
        """
        ret = dict()
        try:
            for i, per_processors_group in enumerate(self.processors):
                for per_modality, per_modality_processors in per_processors_group.items():
                    for per_model_processor in per_modality_processors:
                        ret.update(per_model_processor(getattr(self, f"{per_modality}_{i}"), idx, self.is_training))
        except Exception as e:
            logger.debug(f"Skipping sample {idx} due to '{e}'")
            self._consecutive_errors += 1
            if self._consecutive_errors < GET_ITEM_ERROR_RETRY:
                return self.__getitem__((idx + 1) % self.__len__())
            else:
                raise e
        self._consecutive_errors = 0
        return ret


class EpisodeDataset(BaseDataset):
    """
    The episode dataset for few-shot learning.Each episode includes a few-shot learning task.
    Tensorsupport and labelsupport provide supporting information.
    Tensorquery and labelquery is the part for prediction.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: List[MultiModalFeaturePreprocessor],
        processors: List[dict],
        nCls: int,
        nSupport: int,
        nQuery: int,
        is_training: bool = False,
        nEpisode: int = 2000,
    ):
        """
        Parameters
        ----------
        data
            A pd.DataFrame containing multimodal features.
        preprocessor
            A list of multimodal feature preprocessors generating model-agnostic features.
        processors
            Data processors customizing data for each modality per model.
        nCls
            Number of classes in the few-shot episode.
        nSupport
            Number of samples per class in the support set.
        nQuery
            Number of samples per class in the query set.
        is_training
            Whether in training mode. Some data processing may be different between training
            and validation/testing/prediction, e.g., image data augmentation is used only in
            training.
        nEpisode
            Number of episodes.
        """
        super().__init__(data, preprocessor, processors, is_training)
        self.nCls = nCls
        self.nSupport = nSupport
        self.nQuery = nQuery
        self.nEpisode = nEpisode

        self.clsList = np.array(getattr(self, f"label_0")["Label"].max() + 1)

        self.labelsupport = torch.LongTensor(nCls * nSupport)
        self.labelquery = torch.LongTensor(nCls * nQuery)
        for i in range(self.nCls) :
            self.labelsupport[i * self.nSupport : (i+1) * self.nSupport] = i
            self.labelquery[i * self.nQuery : (i+1) * self.nQuery] = i

    def __len__(self):
        """
        Returns
        -------
        Episode number in this dataset.
        """
        return self.nEpisode

    def __getitem__(self, idx):
        """
        Iterate through all data processors to prepare model inputs. The data processors are
        organized first by modalities and then by models. The data will be sampled for episode
        tasks.

        Parameters
        ----------
        idx
            Index of sample to process.

        Returns
        -------
        Tensorsupport, labelsupport, tensorquery and labelquery formatted as a dictionary.
        """
        clsEpisode = np.random.choice(self.clsList, self.nCls, replace=False)
        support = list()
        query = list()
        for j, cls in enumerate(clsEpisode):
            elelist = np.array([eleidx for eleidx, clsnum in enumerate(getattr(self, f"label_0")["Label"]) if clsnum == cls])
            elecls = np.random.choice(elelist, self.nQuery + self.nSupport, replace=False)
            for k in range(self.nSupport):
                ret = dict()
                try:
                    for i, per_processors_group in enumerate(self.processors):
                        for per_modality, per_modality_processors in per_processors_group.items():
                            for per_model_processor in per_modality_processors:
                                ret.update(per_model_processor(getattr(self, f"{per_modality}_{i}"), elecls[k], self.is_training))
                except Exception as e:
                    logger.debug(f"Skipping sample {elecls[k]} due to '{e}'")
                    self._consecutive_errors += 1
                    if self._consecutive_errors < GET_ITEM_ERROR_RETRY:
                        return self.__getitem__((elecls[k] + 1) % self.__len__())
                    else:
                        raise e
                support.append(ret)
            for k in range(self.nQuery):
                ret = dict()
                try:
                    for i, per_processors_group in enumerate(self.processors):
                        for per_modality, per_modality_processors in per_processors_group.items():
                            for per_model_processor in per_modality_processors:
                                ret.update(per_model_processor(getattr(self, f"{per_modality}_{i}"), elecls[self.nSupport + k], self.is_training))
                except Exception as e:
                    logger.debug(f"Skipping sample {elecls[self.nSupport + k]} due to '{e}'")
                    self._consecutive_errors += 1
                    if self._consecutive_errors < GET_ITEM_ERROR_RETRY:
                        return self.__getitem__((elecls[self.nSupport + k] + 1) % self.__len__())
                    else:
                        raise e
                query.append(ret)
        permSupport = torch.randperm(self.nCls * self.nSupport)
        permQuery = torch.randperm(self.nCls * self.nQuery)
        self.support = [support[perperm] for perperm in permSupport]
        self.query = [query[perperm] for perperm in permQuery]
        self._consecutive_errors = 0
        return {
            "tensorsupport": self.support,
            "labelsupport": self.labelsupport[permSupport],
            "tensorquery": self.query,
            "labelquery": self.labelquery[permQuery],
        }
