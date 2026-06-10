import numpy as np
import config as cfg

# genetic algorithm: breed new birds from the good ones
def make_population(pop_size, n_params, rng):
    # start with totally random brains
    return rng.normal(0, 1, (pop_size, n_params))


def tournament(pop, scores, rng):
    # grab some random birds and return the best of them
    picks = rng.integers(0, len(pop), cfg.tournament_size)
    best = picks[np.argmax(scores[picks])]
    return pop[best]


def pick_parent(pop, scores, order, rng):
    # choose a parent
    if cfg.selection == "topk":
        return pop[order[rng.integers(0, cfg.topk)]]
    return tournament(pop, scores, rng)


def crossover(mom, dad, rng):
    # make kid by taking each weight from a parent at random
    kid = mom.copy()
    for i in range(len(kid)):
        if rng.random() < 0.5:
            kid[i] = dad[i]
    return kid


def mutate(genome, std, rng):
    # randomly change some weights
    kid = genome.copy()
    for i in range(len(kid)):
        if rng.random() < cfg.mutation_rate:
            kid[i] += rng.normal(0, std)
    return kid


def evolve(pop, scores, std, rng):
    # build next generation
    order = np.argsort(scores)[::-1]

    new_pop = []

    # keep the best birds
    for i in range(cfg.elite):
        new_pop.append(pop[order[i]].copy())

    # fill the rest with kids of good parents
    while len(new_pop) < cfg.pop_size:
        mom = pick_parent(pop, scores, order, rng)
        if cfg.crossover:
            dad = pick_parent(pop, scores, order, rng)
            kid = crossover(mom, dad, rng)
        else:
            kid = mom.copy()
        kid = mutate(kid, std, rng)
        new_pop.append(kid)

    return np.array(new_pop)
