"""
This file defines the classes and functions we need for finetuning the DistilBERT transformer.
The code has been adapted from https://github.com/huggingface/transformers/blob/main/examples/tensorflow/text-classification/run_text_classification.py
"""

import json

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from scipy.special import softmax

from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PushToHubCallback,
    TFAutoModelForSequenceClassification,
    create_optimizer,
)

#from transformers.utils import CONFIG_NAME, TF2_WEIGHTS_NAME
import tensorflow as tf 

try:
    import tf_keras as keras
except (ModuleNotFoundError, ImportError):
    import keras

    if int(keras.__version__.split(".")[0]) > 2:
        raise ValueError(
            "Your currently installed version of Keras is Keras 3, but this is not yet supported in "
            "Transformers. Please install the backwards-compatible tf-keras package with "
            "`pip install tf-keras`."
        )

from . import classification_helper as ch

LOGGER = None
DATASETS = {'train': None, 'validation': None, 'test': None}
TF_DATA = {'train': None, 'validation': None, 'test': None}
WRAPPER = None

class SavePretrainedCallback(keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)


class DiseaseClassificationModelWrapper:
    """
    Wrapper class for handling disease classification models.
    """

    def __init__(
            self,
            model_args,
            data_args=None,
            training_args=None,
            tokenizer=None,
            config=None
            ):
        """
        Initializes the wrapper with model and data arguments.

        Args:
            model_args: Model configuration arguments.
            data_args: Data processing arguments.
            training_args: Training arguments.
            tokenizer: Tokenizer object.
            config: Model configuration object.
        """
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code
                )
        else:
            self.tokenizer = tokenizer
        self.config = config
        self.max_seq_length = self.data_args.max_seq_length if data_args else None
        self.label_list = None
        self.num_labels = None
        

    def set_labels(self, ds):
        """
        Sets the labels for the model based on the dataset.

        Args:
            ds: Dataset containing label information.

        Returns:
            None
        """
        if "train" in ds:
            self.label_list = ds["train"].unique("label")
            self.label_list.sort()  # Let's sort it for determinism
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = None
            self.label_list = None

    def set_config(self):
        """
        Sets the model configuration based on the arguments.

        Returns:
            None
        """
        if self.model_args.config_name:
            config_path = self.model_args.config_name
        else:
            config_path = self.model_args.model_name_or_path
        if self.num_labels is not None:
            self.config = AutoConfig.from_pretrained(
                config_path,
                num_labels=self.num_labels,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                token=self.model_args.token,
                trust_remote_code=self.model_args.trust_remote_code,
            )
        else:
            self.config = AutoConfig.from_pretrained(
                config_path,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                token=self.model_args.token,
                trust_remote_code=self.model_args.trust_remote_code,
            )

    def setup(self, ds):
        """
        Setup method to set labels, configuration, and load pretrained model.

        Args:
            ds: Dataset containing label information.

        Returns:
            None
        """
        self.set_labels(ds)
        self.set_config()
        self.load_pretrained_model()

    def load_pretrained_model(self):
        """
        Loads the pretrained model based on the configuration.

        Returns:
            None
        """
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
            )
        if self.config is None:
            self.config = self.model.config

    def preprocess_data(self, ds):
        """
        Preprocesses the dataset by tokenizing texts, mapping labels to IDs, and ensuring consistency between labels and model configurations.

        Args:
            ds: Dataset to be preprocessed.

        Returns:
            ds: Preprocessed dataset.
        """
        
        column_names = {col for cols in ds.column_names.values() for col in cols}
        non_label_column_names = [name for name in column_names if name != "label"]

        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            LOGGER.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(self.data_args.max_seq_length, self.tokenizer.model_max_length)

        # Ensure that our labels match the model's, if it has some pre-specified
        if "train" in ds:
            if self.config.label2id != PretrainedConfig(num_labels=self.num_labels).label2id:
                label_name_to_id = self.config.label2id
                if sorted(label_name_to_id.keys()) == sorted(self.label_list):
                    label_to_id = label_name_to_id  # Use the model's labels
                else:
                    LOGGER.warning(
                        "Your model seems to have been trained with labels, but they don't match the dataset: ",
                        f"model labels: {sorted(label_name_to_id.keys())}, dataset labels:"
                        f" {sorted(self.label_list)}.\nIgnoring the model labels as a result.",
                    )
                    label_to_id = {v: i for i, v in enumerate(self.label_list)}
            else:
                label_to_id = {v: i for i, v in enumerate(self.label_list)}
            # Now we've established our label2id, let's overwrite the model config with it.
            self.config.label2id = label_to_id
            if self.config.label2id is not None:
                self.config.id2label = {id: label for label, id in label_to_id.items()}
            else:
                self.config.id2label = None
        else:
            label_to_id = self.config.label2id  # Just load the data from the model

        if "validation" in ds and self.config.label2id is not None:
            validation_label_list = ds["validation"].unique("label")
            for val_label in validation_label_list:
                assert val_label in label_to_id, f"Label {val_label} is in the validation set but not the training set!"
        
        ds = ds.map(self.preprocess_function, batched=True, load_from_cache_file=not self.data_args.overwrite_cache)
        return ds

    def preprocess_function(self, examples):
        """
        Preprocesses individual examples by tokenizing texts and mapping labels to IDs.

        Args:
            examples: Individual example from the dataset.

        Returns:
            result: Preprocessed example.
        """        
        
        # Tokenize the texts
        args = (
            (examples['sentence1'],)
        )
        result = self.tokenizer(*args, max_length=self.max_seq_length, truncation=True)

        # Map labels to IDs
        if self.config.label2id is not None and "label" in examples:
            result["label"] = [(self.config.label2id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    def convert_to_tf_dataset(self, ds):
        """
        Converts the preprocessed dataset into TensorFlow datasets for training, validation, and testing.

        Args:
            ds: Preprocessed dataset.

        Returns:
            tf_data: TensorFlow datasets for training, validation, and testing.
        """
        
        tf_data = {}
        num_replicas = self.training_args.strategy.num_replicas_in_sync
        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        num_replicas = self.training_args.strategy.num_replicas_in_sync

        for key in ("train", "validation", "test"):
            if key not in ds or not getattr(self.training_args, f"do_{key}", True):
                tf_data[key] = None
                continue

            assert "label" in ds[key].features if key in ("train", "validation") else True, f"Missing labels from {key} data!"

            shuffle = True if key == "train" else False
            batch_size = (self.training_args.per_device_train_batch_size if key == "train" else self.training_args.per_device_eval_batch_size) * num_replicas
            samples_limit = getattr(self.data_args, f"max_{key}_samples", None)

            dataset = ds[key].select(range(samples_limit)) if samples_limit else ds[key]

            data = self.model.prepare_tf_dataset(
                dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                tokenizer=self.tokenizer,
            )

            data = data.with_options(dataset_options)
            tf_data[key] = data

        return tf_data
    
    def prepare_optimizer_loss_compilation(self, tf_data=None):
        """
        Prepares the optimizer and loss functions for model compilation.

        Args:
            tf_data: TensorFlow datasets.

        Returns:
            None
        """
        
        if self.training_args is not None and self.training_args.do_train and tf_data is not None:
            num_train_steps = len(tf_data["train"]) * self.training_args.num_train_epochs
            num_warmup_steps = self.training_args.warmup_steps if self.training_args.warmup_steps > 0 else int(num_train_steps * self.training_args.warmup_ratio) if self.training_args.warmup_ratio > 0 else 0

            optimizer, schedule = create_optimizer(
                init_lr=self.training_args.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                adam_beta1=self.training_args.adam_beta1,
                adam_beta2=self.training_args.adam_beta2,
                adam_epsilon=self.training_args.adam_epsilon,
                weight_decay_rate=self.training_args.weight_decay,
                adam_global_clipnorm=self.training_args.max_grad_norm,
            )
        else:
            optimizer = "sgd"  # Just use any default

        metrics = ["accuracy"]
        self.model.compile(optimizer=optimizer, metrics=metrics)

    def prepare_for_training(self, ds, data_args, training_args):
        """
        Prepares the model, optimizer, and datasets for training.

        Args:
            ds: Dataset for training.
            data_args: Data processing arguments.
            training_args: Training arguments.

        Returns:
            tf_data: TensorFlow datasets for training, validation, and testing.
        """
        
        self.data_args = data_args
        self.training_args = training_args
        tf_data = self.convert_to_tf_dataset(self.preprocess_data(ds))
        self.prepare_optimizer_loss_compilation(tf_data)
        self.prepare_push_to_hub_and_model_card()
        return tf_data

    def prepare_push_to_hub_and_model_card(self):
        """
        Prepares for pushing the fine-tuned model to the Hugging Face Model Hub and creating a model card.

        Returns:
            None
        """
        
        push_to_hub_model_id = self.training_args.push_to_hub_model_id or f"{self.model_args.model_name_or_path.split('/')[-1]}-finetuned-text-classification"
        model_card_kwargs = {"finetuned_from": self.model_args.model_name_or_path, "tasks": "text-classification"}

        self.callbacks = [
            PushToHubCallback(
                output_dir=self.training_args.output_dir,
                hub_model_id=push_to_hub_model_id,
                hub_token=self.training_args.push_to_hub_token,
                tokenizer=self.tokenizer,
                **model_card_kwargs,
            )
        ] if self.training_args.push_to_hub else []
    
    def train(self, train_data, validation_data=None):
        """
        Trains the model.

        Args:
            train_data: Training dataset.
            validation_data: Validation dataset.

        Returns:
            None
        """
        
        if train_data is not None:
            self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=int(self.training_args.num_train_epochs),
                callbacks=self.callbacks,
            )

    def evaluate(self, validation_data=None):
        """
        Evaluates the model.

        Args:
            validation_data: Validation dataset.

        Returns:
            None
        """
        
        if validation_data is not None:
            LOGGER.info("Computing metrics on validation data...")
            loss, accuracy = self.model.evaluate(validation_data)
            LOGGER.info(f"Eval loss: {loss:.5f}, Eval accuracy: {accuracy * 100:.4f}%")

            if self.training_args is not None and self.training_args.output_dir is not None:
                output_eval_file = self.training_args.output_dir + "all_results.json"
                eval_dict = {"eval_loss": loss}
                eval_dict["eval_accuracy"] = accuracy

                with open(output_eval_file, "w") as writer:
                    writer.write(json.dumps(eval_dict))
    
    def train_and_validate(self, train_data, validation_data=None):
        """
        Trains the model and performs evaluation.

        Args:
            train_data: Training dataset.
            validation_data: Validation dataset.

        Returns:
            None
        """
        
        self.train(train_data, validation_data)
        self.evaluate(validation_data)
    
    def predict(self, test_data, output_dir=None):
        """
        Performs predictions on the test dataset.

        Args:
            test_data: Test dataset.
            output_dir: Output directory to save predictions.

        Returns:
            None
        """
        
        if test_data is not None:
            LOGGER.info("Doing predictions on test dataset...")

            predictions = self.model.predict(test_data)["logits"]
            predicted_class = np.argmax(predictions, axis=1)
            if output_dir is None:
                if self.training_args is not None:
                    output_dir = self.training_args.output_dir
                else:
                    output_dir = 'output/'
            output_test_file = output_dir + "test_results.txt"

            with open(output_test_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predicted_class):
                    item = self.config.id2label[item]
                    writer.write(f"{index}\t{item}\n")

            LOGGER.info(f"Wrote predictions to {output_test_file}!")
    

    def predict_sentence(self, sentence):
        """
        Performs prediction on a single sentence.

        Args:
            sentence: Input sentence.

        Returns:
            predictions: Predicted probabilities for each class.
        """
        
        ds = Dataset.from_list([{'sentence1': sentence}])
        ds = ds.map(self.preprocess_function, batched=False, load_from_cache_file=False)
        
        data = self.model.prepare_tf_dataset(
            ds,
            shuffle=False,
            batch_size=1,
            tokenizer=self.tokenizer,
        )

        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        data = data.with_options(dataset_options)

        predictions = self.model.predict(data)["logits"]
        #predicted_class = np.argmax(predictions, axis=1)
        return softmax(predictions)
        #return dict(label_list, predictions)
    
    def save_pretrained(self, output_dir):
        """
        Saves the fine-tuned model.

        Args:
            output_dir: Output directory to save the model.

        Returns:
            None
        """
        
        self.model.save_pretrained(output_dir)


def load_datasets(dataframe=None):
    """
    Load datasets from either a DataFrame or local files specified in the configuration.

    Args:
        dataframe (Optional[pandas.DataFrame]): A DataFrame containing dataset information.

    Returns:
        DatasetDict: A dictionary of datasets loaded from either DataFrame or local files.
    """
    if dataframe is not None:
        df = {}
        for key in dataframe.keys():
            df[key] = Dataset.from_pandas(dataframe[key])
        ds = DatasetDict(df)
        return ds

    data_files = {"train": ch.DATA_ARGS.train_file, "validation": ch.DATA_ARGS.validation_file, "test": ch.DATA_ARGS.test_file}
    data_files = {key: file for key, file in data_files.items() if file is not None}

    for key in data_files.keys():
        LOGGER.info(f"Loading a local file for {key}: {data_files[key]}")

    if ch.DATA_ARGS.input_file_extension == "csv":
        # Loading a dataset from local csv files
        ds = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=ch.MODEL_ARGS.cache_dir,
            token=ch.MODEL_ARGS.token,
        )
    else:
        # Loading a dataset from local json files
        ds = load_dataset("json", data_files=data_files, cache_dir=ch.MODEL_ARGS.cache_dir)
    return ds


def setup_from_scratch(dataframe=None):
    """
    Setup logger, load datasets, setup model wrapper, and prepare data for training.

    Args:
        dataframe (Optional[pandas.DataFrame]): A DataFrame containing dataset information.

    Returns:
        None
    """
    global DATASETS, LOGGER, TF_DATA, WRAPPER

    LOGGER = ch.setup_logging()

    DATASETS = load_datasets(dataframe)

    WRAPPER = DiseaseClassificationModelWrapper(ch.MODEL_ARGS)

    WRAPPER.setup(DATASETS)

    TF_DATA = WRAPPER.prepare_for_training(DATASETS, ch.DATA_ARGS, ch.TRAINING_ARGS)


def train():
    """
    Train the configured model using the prepared training data.

    Returns:
        None
    """
    WRAPPER.train(TF_DATA["train"], TF_DATA["validation"])


def evaluate():
    """
    Evaluate the trained model using the validation data.

    Returns:
        None
    """
    WRAPPER.evaluate(TF_DATA['validation'])


def predict():
    """
    Make predictions using the trained model on the test data.

    Returns:
        None
    """
    WRAPPER.predict(TF_DATA['test'])


def setup_from_finetuned(directory):
    """
    Setup model wrapper from a finetuned model directory.

    Args:
        directory (str): Path to the directory containing the finetuned model.

    Returns:
        None
    """
    model_args = ch.ModelArguments(
        model_name_or_path=directory,
        tokenizer_name="distilbert/distilbert-base-cased",
    )
    WRAPPER = DiseaseClassificationModelWrapper(model_args)
    WRAPPER.load_pretrained_model()
    WRAPPER.prepare_optimizer_loss_compilation()
