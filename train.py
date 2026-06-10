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


def run_evolution(rng, log=True):
    # returns the score history and the best bird
    env = make_env()
    sizes = get_sizes(env)
    net = network(sizes)

    pop = ga.make_population(cfg.pop_size, net.n_params, rng)

    best_list = []
    avg_list = []
    best_genome = None
    best_score = -9999

    for gen in range(cfg.generations):
        # pick the levels everyone plays this generation
        if cfg.eval_seed is None:
            seeds = [int(rng.integers(0, 1000000)) for _ in range(cfg.n_eval)]
        else:
            seeds = [cfg.eval_seed + i for i in range(cfg.n_eval)]

        # score every bird
        scores = np.array([get_score(net, g, env, seeds) for g in pop])

        # remember the best bird
        i = int(np.argmax(scores))
        if scores[i] > best_score:
            best_score = scores[i]
            best_genome = pop[i].copy()

        best_list.append(scores.max())
        avg_list.append(scores.mean())
        if log:
            print("gen", gen, "| best", round(float(scores.max()), 1),
                  "| avg", round(float(scores.mean()), 1))

        # shrink the mutation
        frac = gen / max(cfg.generations - 1, 1)
        std = cfg.mutation_std + frac * (cfg.mutation_std_end - cfg.mutation_std)

        if gen < cfg.generations - 1:
            pop = ga.evolve(pop, scores, std, rng)

    env.close()
    return best_list, avg_list, best_genome, best_score, sizes


def main():
    rng = np.random.default_rng(cfg.seed)
    print("env:", cfg.env_name, "| hidden:", cfg.hidden, "| pop:", cfg.pop_size)

    start = time.time()
    best_list, avg_list, best_genome, best_score, sizes = run_evolution(rng)
    print("done in", round(time.time() - start, 1), "s | best score:", round(best_score, 1))

    # save the best bird
    np.savez(cfg.save_file, genome=best_genome, sizes=np.array(sizes), env_name=cfg.env_name)
    print("saved best bird to", cfg.save_file)

    # plot
    plt.plot(best_list, label="best")
    plt.plot(avg_list, label="average")
    plt.xlabel("generation")
    plt.ylabel("fitness (episode reward)")
    plt.title(f"neuroevolution on {cfg.env_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.plot_file, dpi=120)
    print("saved plot to", cfg.plot_file)


if __name__ == "__main__":
    main()
