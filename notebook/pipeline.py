from pathlib import Path
from surprise import Reader
from surprise.dataset import DatasetAutoFolds, Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.trainset import Trainset
from surprise.prediction_algorithms.algo_base import AlgoBase


def load_ratings_from_file(ratings_filepath: Path) -> DatasetAutoFolds:
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    ratings = Dataset.load_from_file(ratings_filepath, reader)
    return ratings

def load_ratings_from_surprise() -> DatasetAutoFolds:
    ratings = Dataset.load_builtin('ratings.csv')
    return ratings

def get_data(from_surprise: bool = True) -> DatasetAutoFolds:
    data = load_ratings_from_surprise() if from_surprise else load_ratings_from_file()
    return data


def get_model_knn(model_class: AlgoBase, model_kwargs: dict, train_s: Trainset) -> AlgoBase:
    model = model_class(sim_options=model_kwargs)
    model.fit(train_s)
    return model


def get_model_svd(model_class: AlgoBase, train_s: Trainset) -> AlgoBase:
    model = model_class()
    model.fit(train_s)
    return model


def get_model_svdpp(model_class: AlgoBase, train_s: Trainset) -> AlgoBase:
    model = model_class()
    model.fit(train_s)
    return model


def get_model_nmf(model_class: AlgoBase, train_s: Trainset) -> AlgoBase:
    model = model_class()
    model.fit(train_s)
    return model


def evaluate_model(model: AlgoBase, test_s: [(int, int, float)]) -> dict:
    predictions = model.test(test_s)
    metrics_dict = {}
    metrics_dict['RMSE'] = accuracy.rmse(predictions, verbose=False)
    metrics_dict['MAE'] = accuracy.rmse(predictions, verbose=False)
    return metrics_dict


def train_and_evalute_model_pipeline(model_class: AlgoBase, model_kwargs: dict = {},
                                     from_surprise: bool = False,
                                     test_size: float = 0.2) -> (AlgoBase, dict):
    data = get_data(from_surprise)
    train_s, test_s = train_test_split(data, test_size, random_state=42)
    model = get_model_knn(model_class, model_kwargs, train_s)
    metrics_dict = evaluate_model(model, test_s)
    return model, metrics_dict