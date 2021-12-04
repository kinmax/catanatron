import os
import pdb
import json
import time
import random
import sys, traceback
from pathlib import Path
import click

import numpy as np
from tqdm import tqdm
import selenium
from selenium import webdriver

from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron_gym.features import create_sample_vector, get_feature_ordering
from catanatron_server.utils import ensure_link
from catanatron_gym.envs.catanatron_env import (
    from_action_space,
    to_action_space,
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
)
from catanatron.state_functions import (
    player_key,
)


FEATURES = get_feature_ordering(2)
NUM_FEATURES = len(FEATURES)

DISCOUNT = 0.9

# Environment exploration settings
# tutorial settings (seems like 26 hours...)
# EPISODES = 20_000
# EPSILON_DECAY = 0.99975
# ALPHA_DECAY = 0.99975
# 8 hours process
# EPISODES = 6000
# EPSILON_DECAY = 0.9993
# ALPHA_DECAY = 0.9993
# 2 hours process
# EPISODES = 1500
# EPSILON_DECAY = 0.998
# ALPHA_DECAY = 0.998
# 30 mins process
EPISODES = 500
EPSILON_DECAY = 0.994
ALPHA_DECAY = 0.994
# EPISODES = 10_000
epsilon = 1  # not a constant, going to be decayed
alpha = 1
MIN_EPSILON = 0.001
MIN_ALPHA = 0.001

# Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = False

# TODO: Simple Action Space:
# Hold
# Build Settlement on most production spot vs diff. need number to translate enemy potential to true prod.
# Build City on most production spot.
# Build City on spot that balances production the most.
# Build Road towards more production. (again need to translate potential to true.)
# Buy dev card
# Play Knight to most powerful spot.
# Play Year of Plenty towards most valueable play (city, settlement, dev). Bonus points if use rare resources.
# Play Road Building towards most increase in production.
# Play Monopoly most impactful resource.
# Trade towards most valuable play.

# TODO: Simple State Space:
# Cards in Hand
# Buildable Nodes
# Production
# Num Knights
# Num Roads

DATA_PATH = "data/mcts-playouts"
NORMALIZATION_MEAN_PATH = Path(DATA_PATH, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(DATA_PATH, "variance.npy")


class CatanEnvironment:
    def __init__(self):
        self.game = None
        self.p0 = None

    def playable_actions(self):
        return self.game.state.playable_actions

    def reset(self):
        p0 = Player(Color.RED)
        players = [p0, VictoryPointPlayer(Color.BLUE)]
        game = Game(players=players)
        self.game = game
        self.p0 = p0

        self._advance_until_p0_decision()

        return self._get_state()

    def step(self, action_int):
        key = player_key(self.game.state, self.p0.color)
        old_points = self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        enemy_key = player_key(self.game.state, Color.BLUE)
        enemy_old_points = self.game.state.player_state[f"{enemy_key}_VICTORY_POINTS"]
        
        try:
            action = from_action_space(action_int, self.playable_actions())
        except:
            a = 5
            pdb.set_trace()
        self.game.execute(action)

        self._advance_until_p0_decision()
        winning_color = self.game.winning_color()

        new_state = self._get_state()

        key = player_key(self.game.state, self.p0.color)
        points = self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        enemy_key = player_key(self.game.state, Color.BLUE)
        enemy_points = self.game.state.player_state[f"{enemy_key}_VICTORY_POINTS"]
        if winning_color is None:
            final_reward = 0
        elif winning_color == self.p0.color:
            final_reward = 10
        else:
            final_reward = -10
        reward = final_reward + (points - old_points) - (enemy_points - enemy_old_points)
        # if winning_color is None:
        #     reward = 0
        # elif winning_color == self.p0.color:
        #     reward = 1
        # else:
        #     reward = -1

        done = winning_color is not None or self.game.state.num_turns > 500
        return new_state, reward, done

    def render(self):
        driver = webdriver.Chrome()
        link = ensure_link(self.game)
        driver.get(link)
        time.sleep(1)
        try:
            driver.close()
        except selenium.common.exceptions.WebDriverException as e:
            print("Exception closing browser. Did you close manually?")

    def _get_state(self):
        sample = create_sample_vector(self.game, self.p0.color, FEATURES)

        return sample  # NOTE: each observation/state is a tuple.

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_player().color != self.p0.color
        ):
            self.game.play_tick()  # will play bot


def epsilon_greedy_policy(playable_actions, qs, epsilon):
    if np.random.random() > epsilon:
        # Create array like [0,0,1,0,0,0,1,...] representing actions in space that are playable
        action_ints = list(map(to_action_space, playable_actions))
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int)
        mask[action_ints] = 1

        clipped_probas = qs.copy()
        for i in range(0, len(clipped_probas)):
            if i not in action_ints:
                clipped_probas[i] = -np.inf


        # clipped_probas = np.multiply(mask, qs)
        # clipped_probas[clipped_probas == 0] = -np.inf

        best_action_int = np.argmax(clipped_probas)

    else:
        # Get random action
        index = random.randrange(0, len(playable_actions))
        best_action = playable_actions[index]
        best_action_int = to_action_space(best_action)

    return best_action_int



class QLPlayer(Player):
    def __init__(self, color, model_name):
        super(QLPlayer, self).__init__(color)
        self.model_path = f"./data/tables/ql-player/{model_name}/{model_name}.json"
        data = ""
        with open(self.model_path, 'r') as file:
            data = file.read()
        q_table_dict = json.loads(data)
        self.q_table = {}
        for key, value in q_table_dict.items():
            self.q_table[key] = np.array(value)
        self.metrics_path = f"data/metrics/ql-player/{model_name}/benchmark_metrics.csv"
        self.known_states = 0
        self.unknown_states = 0
        self.total_states = 0
        self.game_counter = 0
        self.wins = 0
        self.ties = 0
        self.losses = 0
        with open(self.metrics_path, "a") as text_file:
            print("game,win,tie,loss,avg_seen_states,avg_known_states,avg_unknown_states,q_table_size", file=text_file)
        # print("INIT")

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        sample = create_sample_vector(game, self.color, FEATURES)
        self.total_states += 1
        if repr(sample) in self.q_table:
            # print("Q-TABLE")
            self.known_states += 1
            qs = self.q_table[repr(sample)]
            e = 0.0
        else:
            # print("RANDOM")
            self.unknown_states += 1
            qs = np.zeros(ACTION_SPACE_SIZE)
            e = 1.0

        best_action_int = epsilon_greedy_policy(playable_actions, qs, e)
        best_action = from_action_space(best_action_int, playable_actions)
        return best_action

    def log_data(self, win):
        self.game_counter += 1
        if win == 1:
            self.wins += 1
        elif win == 0:
            self.ties += 1
        else:
            self.losses += 1
        
    def save_data(self):
        data = f"{self.game_counter},{self.wins},{self.ties},{self.losses},{self.total_states/self.game_counter},{self.known_states/self.game_counter},{self.unknown_states/self.game_counter},{len(list(self.q_table.keys()))}"
        with open(self.metrics_path, "a") as text_file:
            print(data, file=text_file)
        


@click.command()
@click.argument("experiment_name")
@click.argument("gamma")
def main(experiment_name, gamma):
    global epsilon, alpha

    q_table = {}
    gamma = float(gamma)

    env = CatanEnvironment()

    # For stats
    ep_rewards = []

    # For more repetitive results
    random.seed(2)
    np.random.seed(2)

    # Ensure models folder
    model_name = f"{experiment_name}" #-{int(time.time())}"
    models_folder = f"data/tables/ql-player/{model_name}/"
    if not os.path.isdir(models_folder):
        os.makedirs(models_folder)

    metrics_path = f"data/metrics/ql-player/{model_name}/"
    if not os.path.isdir(metrics_path):
        os.makedirs(metrics_path)

    metrics_path += "metrics.txt"

    with open(metrics_path, "w") as text_file:
        print("episodes,average_reward,epsilon,alpha,q_table_state_count", file=text_file)

    output_model_path = models_folder + model_name + ".json"
    print("Will be saving metrics to", metrics_path)
    print("Will be saving Q-Table to", output_model_path)

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()
        if not repr(current_state) in q_table:
            q_table[repr(current_state)] = np.empty(ACTION_SPACE_SIZE)
            q_table[repr(current_state)].fill(np.random.random())

        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            best_action_int = epsilon_greedy_policy(
                env.playable_actions(), q_table[repr(current_state)], epsilon
            )

            try:            
                new_state, reward, done = env.step(best_action_int)
            except:
                pdb.set_trace()
            

            if not repr(new_state) in q_table:
                # print("NOT IN TABLE")
                if not done:
                    q_table[repr(new_state)] = np.empty(ACTION_SPACE_SIZE)
                    q_table[repr(new_state)].fill(np.random.random())
                else:
                    q_table[repr(new_state)] = np.zeros(ACTION_SPACE_SIZE, dtype=np.float)
            else:
                x = 5
                # print("FOUND IT")


            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

            q_table[repr(current_state)][best_action_int] = q_table[repr(current_state)][best_action_int] + alpha * (reward + (gamma * q_table[repr(new_state)][np.argmax(q_table[repr(new_state)])]) - q_table[repr(current_state)][best_action_int])

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if episode % AGGREGATE_STATS_EVERY == 0:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
                ep_rewards[-AGGREGATE_STATS_EVERY:]
            )
            with open(metrics_path, "a") as text_file:
                print(f"{episode},{average_reward},{epsilon},{alpha},{len(list(q_table.keys()))}", file=text_file)

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        if alpha > MIN_ALPHA:
            alpha *= ALPHA_DECAY
            alpha = max(MIN_ALPHA, alpha)

    print("Saving Q-Table to", output_model_path)
    serializable_q_table = {}
    for key, value in q_table.items():
        serializable_q_table[key] = value.tolist()
    with open(output_model_path, "w") as text_file:
        print(json.dumps(serializable_q_table, indent = 2), file=text_file)


if __name__ == "__main__":
    main()
