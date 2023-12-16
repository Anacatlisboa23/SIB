from typing import Dict, Tuple, Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.cross_validation import k_fold_cross_validation

def randomized_search_cv(model,
                         validation_dataset: Dataset,
                         param_distribution: Dict[str, Tuple],
                         scoring_function: Callable = None,
                         num_folds: int = 3,
                         num_iterations: int = 10,
                         test_set_size: float = 0.3) -> Dict[str, Tuple[str, Union[int, float]]]:
    """
    Optimize model parameters using random combinations.
    Evaluates a random set of parameters drawn from a distribution or set of possible values.
    More efficient than grid search and may find better hyperparameter values.

    :param model: Model to validate
    :param validation_dataset: Validation dataset
    :param param_distribution: Parameters for the search. Dictionary with parameter names and value distributions.
    :param scoring_function: Scoring function
    :param num_folds: Number of folds for cross-validation
    :param num_iterations: Number of random parameter combinations
    :param test_set_size: Size of the test dataset

    :return: Dictionary with parameter combinations and corresponding training and test scores
    """
    scores = {'parameters': [], 'seed': [], 'train': [], 'test': []}

    # Check if parameters exist in the model
    for parameter in param_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"The model does not have parameter {parameter}.")

    # Set num_iterations parameter combinations
    for i in range(num_iterations):

        # Set the random seed
        random_state = np.random.randint(0, 1000)

        # Store the seed
        scores['seed'].append(random_state)

        # Dictionary for the parameter configuration
        parameters = {}

        # Select the parameters and their values
        for parameter, value_range in param_distribution.items():
            parameters[parameter] = np.random.uniform(low=value_range[0], high=value_range[1])

        # Set the parameters in the model
        for parameter, value in parameters.items():
            setattr(model, parameter, value)

        # Get scores from cross-validation
        score = k_fold_cross_validation(model=model, dataset=validation_dataset, scoring=scoring_function, cv=num_folds)

        # Store the parameter combination and the obtained score in the dictionary
        scores['parameters'].append(parameters)
        scores['train'].append(score)
        scores['test'].append(score)

    return scores


if __name__ == '__main__':
    # Imports
    from si.io.csv_file import read_csv
    from sklearn.preprocessing import StandardScaler
    from si.models.logistic_regression import LogisticRegression

    # Read and standardize the dataset
    breast_bin_path = r"C:\Users\pc\PycharmProjects\si\datasets\breast_bin\breast-bin.data"
    dataset = read_csv(breast_bin_path, sep=",", label=True)
    dataset.X = StandardScaler().fit_transform(dataset.X)

    # Initialize the randomized search
    lr_model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    lr_model_param = {'l2_penalty': np.linspace(1, 10, 10),
                      'alpha': np.linspace(0.001, 0.0001, 100),
                      'max_iter': np.linspace(1000, 2000, 200)}
    scores = randomized_search_cv(lr_model, dataset, lr_model_param, num_folds=3)
    print(f'Scores: ', scores)
