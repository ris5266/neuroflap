import numpy as np

# brain of the bird
class network:
    def __init__(self, sizes):
        self.sizes = sizes # [12, 8, 2] -> 12 in, 8 hidden, 2 out

        # count how many numbers it needs for all the weights and biases
        self.n_params = 0
        for i in range(len(sizes) - 1):
            self.n_params += sizes[i] * sizes[i + 1] # weights
            self.n_params += sizes[i + 1] # biases

    def act(self, genome, obs):
        x = np.array(obs)
        pos = 0

        for i in range(len(self.sizes) - 1):
            n_in = self.sizes[i]
            n_out = self.sizes[i + 1]

            # cut out weights
            w = genome[pos:pos + n_in * n_out].reshape(n_in, n_out)
            pos += n_in * n_out

            # cut out biases
            b = genome[pos:pos + n_out]
            pos += n_out

            x = x @ w + b
            if i < len(self.sizes) - 2:
                x = np.tanh(x)

        return int(np.argmax(x))
