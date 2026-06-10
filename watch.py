import numpy as np
from network import network
from evaluate import make_env, play_game

# load the best trained model
data = np.load("best.npz", allow_pickle=True)
genome = data["genome"]
sizes = list(data["sizes"]) if "sizes" in data else list(data["layer_sizes"])

net = network(sizes)
env = make_env(render="human")

for i in range(3):
    score = play_game(net, genome, env, seed=None)   # random level each time
    print("game", i + 1, "| score:", round(score, 1))

env.close()
