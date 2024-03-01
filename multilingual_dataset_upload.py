# coding=utf-8

import os
import datasets

_CITATION = """\
@InProceedings{Patient-DoctorTextClassifier,
 author = {Luca Flammia},
 title = {Patient - Doctor Text Classifier},
 year = {2024}}
"""

_DESCRIPTION = """\
 Patient - Doctor Text Classifier in Social Media Contents.
 This is a dataset for text classification for social media contents.
 'Given a social media content, classify it as 'neutral or no emotion' or as one, or more, of eleven given emotions that best represent the mental state of the tweeter.'
 It contains 22467 tweets in three languages manually annotated by crowdworkers using Best–Worst Scaling.
"""

_HOMEPAGE = "https://www.pulsarplatform.com/"

_LICENSE = ""

_URLs = {
  "english": ["https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"],
  # "italian": ["https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"],
}


class PatientDoctorTextClassifier(datasets.GeneratorBasedBuilder):

  VERSION = datasets.Version("1.18.2")

  BUILDER_CONFIGS = [
    datasets.BuilderConfig(
      name="english",
      version=VERSION,
      description="This is the English dataset of Patient - Doctor Text Classifier.",
    ),
    # datasets.BuilderConfig(
    #   name="italian",
    #   version=VERSION,
    #   description="This is the Italian dataset of Patient - Doctor Text Classifier.",
    # ),
  ]

  def _info(self):
    features = datasets.Features(
      {
        "text": datasets.Value("string"),
        "label": datasets.Value("int64"),
      }
    )

    return datasets.DatasetInfo(
      description=_DESCRIPTION,
      features=features,
      supervised_keys=None,
      homepage=_HOMEPAGE,
      license=_LICENSE,
      citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    my_urls = _URLs[self.config.name]
    if self.config.name == "english":
      shortname = "En"
    # if self.config.name == "italian":
    #   shortname = "It"
    data_dir = dl_manager.download_and_extract(my_urls)
    return [
      datasets.SplitGenerator(
        name=datasets.Split.TRAIN,
        gen_kwargs={
          "filepath": os.path.join(
            data_dir[0],
            "Patient-Doctor-Text-Classifier-all-data/" + shortname + "-train.txt",
          ),
          "split": "train",
        },
      ),
      datasets.SplitGenerator(
        name=datasets.Split.TEST,
        gen_kwargs={
          "filepath": os.path.join(
            data_dir[0],
            "Patient-Doctor-Text-Classifier-all-data/" + shortname + "-test-gold.txt",
          ),
          "split": "test",
        },
      ),
      datasets.SplitGenerator(
        name=datasets.Split.VALIDATION,
        gen_kwargs={
          "filepath": os.path.join(
            data_dir[0],
            "Patient-Doctor-Text-Classifier-all-data/" + shortname + "-val.txt",
          ),
          "split": "val",
        },
      ),
    ]

  def _generate_examples(self, filepath, split):
      """Yields examples as (key, example) tuples."""

      with open(filepath, encoding="utf-8") as f:
        next(f)  # skip header
        for id_, row in enumerate(f):
          data = row.split("\t")
          yield id_, {
            "text": data[0],
            "label": data[1],
          }