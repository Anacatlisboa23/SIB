from typing import Dict, Tuple, Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.cross_validation import k_fold_cross_validation

def randomized_search_cv(model,
                         dataset: Dataset,
                         parameter_distribution: Dict[str, Tuple],
                         scoring: Callable = None,
                         cv: int = 3,
                         n_iter: int = 10,
                         test_size: float = 0.3) -> Dict[str, Tuple[str, Union[int, float]]]:
    """
    Method to optimize parameters using random combinations.
    Evaluates only a random set of parameters drawn from a distribution or set of possible values.
    It is more efficient than grid search and can find better hyperparameter values.

    :param model: Model to validate
    :param dataset: Validation dataset
    :param parameter_distribution: Parameters for the search. Dictionary with parameter names and value distributions.
    :param scoring: Scoring function
    :param cv: Number of folds
    :param n_iter: Number of random parameter combinations
    :param test_size: Size of the test dataset

    :return: List of dictionaries with parameter combinations and training and test scores
    """
    scores = {'parameters': [], 'seed': [], 'train': [], 'test': []}

    # checks if parameters exist in the model
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"The {model} does not have parameter {parameter}.")

    # sets n_iter parameter combinations
    for i in range(n_iter):

        # set the random seed
        random_state = np.random.randint(0, 1000)

        # store the seed
        scores['seed'].append(random_state)

        # dictionary for the parameter configuration
        parameters = {}

        # select the parameters and its value
        for parameter, value in parameter_distribution.items():
            parameters[parameter] = np.random.uniform(low=value[0], high=value[1])

        # set the parameters to the model
        for parameter, value in parameters.items():
            setattr(model, parameter, value)

        # get scores from cross_validation
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # stores the parameter combination and the obtained score in the dictionary
        scores['parameters'].append(parameters)
        scores['train'].append(score)
        scores['test'].append(score)

    return scores


if __name__ == '__main__':
    # imports
    from si.io.csv_file import read_csv
    from sklearn.preprocessing import StandardScaler
    from si.models.logistic_regression import LogisticRegression

    # read and standardize the dataset
    breast_bin = r"C:\Users\pc\PycharmProjects\si\datasets\breast_bin\breast-bin.data"
    dataset = read_csv(breast_bin, sep=",", label=True)
    dataset.X = StandardScaler().fit_transform(dataset.X)

    # initialize the randomized search
    lg_model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    lg_model_param = {'l2_penalty': np.linspace(1, 10, 10),
                      'alpha': np.linspace(0.001, 0.0001, 100),
                      'max_iter': np.linspace(1000, 2000, 200)}
    scores = randomized_search_cv(lg_model, dataset, lg_model_param, cv=3)
    print(f'Scores: ', scores)
