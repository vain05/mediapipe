# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Specifications for text classifier models."""

import dataclasses
import enum
import functools

from mediapipe.model_maker.python.core.utils import file_util
from mediapipe.model_maker.python.text.core import bert_model_spec
from mediapipe.model_maker.python.text.text_classifier import hyperparameters as hp
from mediapipe.model_maker.python.text.text_classifier import model_options as mo


MOBILEBERT_TINY_FILES = file_util.DownloadedFiles(
    'text_classifier/mobilebert_tiny',
    'https://storage.googleapis.com/mediapipe-assets/mobilebert_tiny.tar.gz',
    is_folder=True,
)

EXBERT_FILES = file_util.DownloadedFiles(
    'text_classifier/exbert',
    'https://storage.googleapis.com/mediapipe-assets/exbert.tar.gz',
    is_folder=True,
)


@dataclasses.dataclass
class AverageWordEmbeddingClassifierSpec:
  """Specification for an average word embedding classifier model.

  Attributes:
    hparams: Configurable hyperparameters for training.
    model_options: Configurable options for the average word embedding model.
    name: The name of the object.
  """

  # `learning_rate` is unused for the average word embedding model
  hparams: hp.AverageWordEmbeddingHParams = hp.AverageWordEmbeddingHParams(
      epochs=10, batch_size=32, learning_rate=0
  )
  model_options: mo.AverageWordEmbeddingModelOptions = (
      mo.AverageWordEmbeddingModelOptions())
  name: str = 'AverageWordEmbedding'

average_word_embedding_classifier_spec = functools.partial(
    AverageWordEmbeddingClassifierSpec)


@dataclasses.dataclass
class BertClassifierSpec(bert_model_spec.BertModelSpec):
  """Specification for a Bert classifier model.

  Only overrides the hparams attribute since the rest of the attributes are
  inherited from the BertModelSpec.
  """

  hparams: hp.BertHParams = hp.BertHParams()


mobilebert_classifier_spec = functools.partial(
    BertClassifierSpec,
    downloaded_files=MOBILEBERT_TINY_FILES,
    hparams=hp.BertHParams(
        epochs=3, batch_size=48, learning_rate=3e-5, distribution_strategy='off'
    ),
    name='MobileBert',
    tflite_input_name={
        'ids': 'serving_default_input_1:0',
        'segment_ids': 'serving_default_input_2:0',
        'mask': 'serving_default_input_3:0',
    },
)

exbert_classifier_spec = functools.partial(
    BertClassifierSpec,
    downloaded_files=EXBERT_FILES,
    hparams=hp.BertHParams(
        epochs=3, batch_size=48, learning_rate=3e-5, distribution_strategy='off'
    ),
    name='ExBert',
    tflite_input_name={
        'ids': 'serving_default_input_1:0',
        'segment_ids': 'serving_default_input_2:0',
        'mask': 'serving_default_input_3:0',
    },
)


@enum.unique
class SupportedModels(enum.Enum):
  """Predefined text classifier model specs supported by Model Maker."""
  AVERAGE_WORD_EMBEDDING_CLASSIFIER = average_word_embedding_classifier_spec
  MOBILEBERT_CLASSIFIER = mobilebert_classifier_spec
  EXBERT_CLASSIFIER = exbert_classifier_spec
