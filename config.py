env_name = "FlappyBird-v0"
max_steps = 8000

# network
hidden = 8                   # neurons in the hidden layer

# evolution settings
pop_size = 100               # how many birds in the population
generations = 200            # how many rounds of evolution
elite = 4                    # how many best birds keep unchanged
tournament_size = 7          # how many birds fight to be a parent
n_eval = 5                   # how many games each bird plays (averaged)

crossover = False            # mix two parents?
mutation_rate = 0.2          # chance each weight gets changed
mutation_std = 0.3           # how big the change is at the start
mutation_std_end = 0.03      # mutation shrinks down to this at the end

eval_seed = None             # random levels each generation
seed = 42                    # random seed

save_file = "best.npz"
plot_file = "fitness.png"
