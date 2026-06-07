import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from network import network
from evaluate import make_env, get_score
import ga

def get_sizes(env):
    n_in = int(env.observation_space.shape[0])
    n_out = int(env.action_space.n)
    return [n_in, cfg.hidden, n_out]


def main():
    rng = np.random.default_rng(cfg.seed)

    env = make_env()
    sizes = get_sizes(env)
    net = network(sizes)
    print("env:", cfg.env_name, "| network:", sizes, "| weights:", net.n_params)

    pop = ga.make_population(cfg.pop_size, net.n_params, rng)

    best_list = []
    avg_list = []
    best_genome = None
    best_score = -9999
    start = time.time()

    for gen in range(cfg.generations):
        if cfg.eval_seed is None:
            seeds = [int(rng.integers(0, 1000000)) for _ in range(cfg.n_eval)]
        else:
            seeds = [cfg.eval_seed + i for i in range(cfg.n_eval)]

        scores = np.array([get_score(net, g, env, seeds) for g in pop])

        # remember the best bird
        i = int(np.argmax(scores))
        if scores[i] > best_score:
            best_score = scores[i]
            best_genome = pop[i].copy()

        best_list.append(scores.max())
        avg_list.append(scores.mean())
        print("gen", gen, "| best", round(float(scores.max()), 1), "| avg", round(float(scores.mean()), 1))

        frac = gen / max(cfg.generations - 1, 1)
        std = cfg.mutation_std + frac * (cfg.mutation_std_end - cfg.mutation_std)

        if gen < cfg.generations - 1:
            pop = ga.evolve(pop, scores, std, rng)

    env.close()
    print("done in", round(time.time() - start, 1), "s | best score:", round(best_score, 1))

    # save the best bird
    np.savez(cfg.save_file, genome=best_genome, sizes=np.array(sizes), env_name=cfg.env_name)
    print("saved best bird to", cfg.save_file)

    # plot
    plt.plot(best_list, label="best")
    plt.plot(avg_list, label="average")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig(cfg.plot_file)
    print("saved plot to", cfg.plot_file)


main()
