import gymnasium as gym
import flappy_bird_gymnasium
import config as cfg

# run the bird and gives back its score
def make_env(render=None):
    if cfg.env_name.startswith("FlappyBird"):
        return gym.make(cfg.env_name, render_mode=render, use_lidar=False)
    return gym.make(cfg.env_name, render_mode=render)


def play_game(net, genome, env, seed=None):
    # play one full game and add up all the reward
    obs, _ = env.reset(seed=seed)
    total = 0
    for _ in range(cfg.max_steps):
        action = net.act(genome, obs)
        obs, reward, done, trunc, info = env.step(action)
        total += reward
        if done or trunc:
            break
    return total


def get_score(net, genome, env, seeds):
    # play a few different levels and average them
    scores = [play_game(net, genome, env, s) for s in seeds]
    return sum(scores) / len(scores)
