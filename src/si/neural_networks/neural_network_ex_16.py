from si.neural_networks.optimizers import Optimizer
from si.metrics import accuracy

class NeuralNetwork:
    def __init__(self, optimizer: Optimizer, loss, metric, epochs, batch_size, verbose=False, **kwargs):
        self.layers = []
        self.history = {'loss': [], 'metric': []}
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.kwargs = kwargs

    def add(self, layer):
        if not self.layers:
            if layer.input_shape() is None:
                raise ValueError("Input shape must be specified for the first layer.")
        else:
            layer.set_input_shape(self.layers[-1].output_shape())
        layer.initialize(self.optimizer, **self.kwargs)
        self.layers.append(layer)

    def fit(self, dataset):
        for epoch in range(self.epochs):
            for data_batch, labels_batch in dataset.batch_generator(self.batch_size):
                self.train_on_batch(data_batch, labels_batch)
            self.evaluate(dataset)

    def train_on_batch(self, data_batch, labels_batch):
        # Forward propagation
        output = data_batch
        for layer in self.layers:
            output = layer.forward_propagation(output, training=True)

        # Backward propagation
        error = self.loss.derivative(labels_batch, output)
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)

        # Update weights and biases
        for layer in self.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                layer.weights = self.optimizer.update(layer.weights, layer.w_opt.m)
                layer.biases = self.optimizer.update(layer.biases, layer.b_opt.m)

    def evaluate(self, dataset):
        total_loss = 0.0
        total_metric = 0.0
        total_samples = 0

        for data_batch, labels_batch in dataset.batch_generator(self.batch_size):
            output = data_batch
            for layer in self.layers:
                output = layer.forward_propagation(output, training=False)

            total_loss += self.loss.loss(labels_batch, output)
            total_metric += self.metric(labels_batch, output)
            total_samples += len(data_batch)

        average_loss = total_loss / total_samples
        average_metric = total_metric / total_samples

        self.history['loss'].append(average_loss)
        self.history['metric'].append(average_metric)

        if self.verbose:
            print(f'Epoch: {len(self.history["loss"])}, Loss: {average_loss}, Metric: {average_metric}')

    def predict(self, data):
        output = data
        for layer in self.layers:
            output = layer.forward_propagation(output, training=False)
        return output

    def score(self, dataset):
        total_metric = 0.0
        total_samples = 0

        for data_batch, labels_batch in dataset.batch_generator(self.batch_size):
            output = self.predict(data_batch)
            total_metric += self.metric(labels_batch, output)
            total_samples += len(data_batch)

        return total_metric / total_samples
