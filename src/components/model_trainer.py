from typing import Tuple, Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def train_and_evaluate_model(self, model, x_train, y_train, x_test, y_test, model_name: str) -> Dict[str, float]:
        """
        Train and evaluate a given model.
        :param model: The model to train.
        :param x_train: Training features.
        :param y_train: Training labels.
        :param x_test: Test features.
        :param y_test: Test labels.
        :param model_name: Name of the model (for logging purposes).
        :return: Dictionary of metrics (accuracy, f1, precision, recall).
        """
        logging.info(f"Training {model_name}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric(f"{model_name}_accuracy", accuracy)
        mlflow.log_metric(f"{model_name}_f1_score", f1)
        mlflow.log_metric(f"{model_name}_precision", precision)
        mlflow.log_metric(f"{model_name}_recall", recall)

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains multiple models, evaluates them, and selects the best one.

        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training and evaluating multiple models...")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("Train-test split done.")

            # Initialize models
            models = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=self.model_trainer_config._n_estimators,
                    min_samples_split=self.model_trainer_config._min_samples_split,
                    min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                    max_depth=self.model_trainer_config._max_depth,
                    criterion=self.model_trainer_config._criterion,
                    random_state=self.model_trainer_config._random_state
                ),
                "LogisticRegression": LogisticRegression(
                    max_iter=1000,
                    random_state=self.model_trainer_config._random_state
                ),
                "GradientBoosting": GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.model_trainer_config._random_state
                )
            }

            # Start MLflow run
            with mlflow.start_run():
                best_model = None
                best_metrics = {"accuracy": 0, "f1_score": 0, "precision": 0, "recall": 0}
                best_model_name = ""

                # Train and evaluate each model
                for model_name, model in models.items():
                    metrics = self.train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name)

                    # Check if the current model is the best
                    if metrics["accuracy"] > best_metrics["accuracy"]:
                        best_model = model
                        best_metrics = metrics
                        best_model_name = model_name

                # Log the best model's metrics
                mlflow.log_metric("best_model_accuracy", best_metrics["accuracy"])
                mlflow.log_metric("best_model_f1_score", best_metrics["f1_score"])
                mlflow.log_metric("best_model_precision", best_metrics["precision"])
                mlflow.log_metric("best_model_recall", best_metrics["recall"])
                mlflow.log_param("best_model", best_model_name)

                # Log the best model
                mlflow.sklearn.log_model(best_model, "best_model")

                # Creating metric artifact
                metric_artifact = ClassificationMetricArtifact(
                    f1_score=best_metrics["f1_score"],
                    precision_score=best_metrics["precision"],
                    recall_score=best_metrics["recall"]
                )
                return best_model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps

        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Train-test data loaded")

            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")

            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performance is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e