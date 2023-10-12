
def manhattan_distance(x, y):
    """
    Calculate the Manhattan distance between a single sample x and multiple samples in y.

    Parameters:
    x (list or tuple): A single sample.
    y (list of lists or tuples): Multiple samples.

    Returns:
    list: An array containing the Manhattan distances between x and the various samples in y.
    """
    distances = []

    for sample in y:
        distance = sum(abs(x_i - y_i) for x_i, y_i in zip(x, sample))
        distances.append(distance)

    return distances

# Examplo:
if __name__ == "__main__":
    x = [1, 2, 3]
    y = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
    result = manhattan_distance(x, y)
    print(result)
