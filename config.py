env_name = "FlappyBird-v0"
max_steps = 8000             # max frames in one game

# network
hidden = 8                   # neurons in the hidden layer

# evolution settings
pop_size = 120               # how many birds in the population
generations = 300            # how many rounds of evolution
elite = 5                    # how many best birds we keep unchanged

# selection
selection = "tournament"
tournament_size = 7          # how many birds fight to be a parent
topk = 15                    # parents are picked from the best this many

# how we score a bird
n_eval = 8                   # how many games each bird plays
eval_agg = "mean"            # "mean" = average skill, "min" = worst game
eval_seed = 0

# making babies
crossover = False            # mix two parents?
mutation_rate = 0.2          # chance each weight gets changed
mutation_std = 0.3           # how big the change is at the start
mutation_std_end = 0.02      # mutation shrinks down to this at the end

seed = 42

save_file = "best.npz"
plot_file = "fitness.png"
