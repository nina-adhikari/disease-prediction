from transformers import TFTrainingArguments
import logging
from dataclasses import dataclass, field
from typing import Optional



DIRECTORY = ''

def setup_logging():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    return logger



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. "
                "Data will always be padded when using TPUs."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of test examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        train_extension = self.train_file.split(".")[-1].lower() if self.train_file is not None else None
        validation_extension = (
            self.validation_file.split(".")[-1].lower() if self.validation_file is not None else None
        )
        test_extension = self.test_file.split(".")[-1].lower() if self.test_file is not None else None
        extensions = {train_extension, validation_extension, test_extension}
        extensions.discard(None)
        assert len(extensions) != 0, "Need to supply at least one of --train_file, --validation_file or --test_file!"
        assert len(extensions) == 1, "All input files should have the same file extension, either csv or json!"
        assert "csv" in extensions or "json" in extensions, "Input files should have either .csv or .json extensions!"
        self.input_file_extension = extensions.pop()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )



TRAINING_ARGS = TFTrainingArguments(
    output_dir=DIRECTORY+'output',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=False,
    num_train_epochs=1
)
MODEL_ARGS = ModelArguments(
    model_name_or_path="distilbert/distilbert-base-cased",
    tokenizer_name="distilbert/distilbert-base-cased",
)
DATA_ARGS = DataTrainingArguments(
    train_file= DIRECTORY+"text-train.json",
    validation_file= DIRECTORY+"text-validate.json",
    test_file= DIRECTORY+"text-test.json",
    #max_seq_length=128,
    overwrite_cache=True,
    max_train_samples=10000,
    max_val_samples=1000,
    max_test_samples=1000,
)