import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="zombie", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="default", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy_snapshots/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=50, help="save model once every time this many episodes are completed")
    parser.add_argument("--display-rate", type=int, default=50, help="display episode once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    return env

def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist)
        n = len(env.agents)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(n)]
        trainers = []
        for i in range(n):
            trainers.append(
                MADDPGAgentTrainer(
                    "agent_%d" % i,
                    mlp_model,
                    obs_shape_n,
                    env.action_space,
                    i,
                    arglist,
                    local_q_func=False
                )
            )

        saver = tf.train.Saver(max_to_keep=None)

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore:
            print('Loading previous state...')
            saver.restore(U.get_session(), arglist.load_dir)

        rewards = np.zeros((1, n))  # agent reward per step
        obs_n = env.reset()
        episode_number = 0
        episode_step = 0
        train_step = 0
        t_start = time.time()

        # stats buffers
        step_info = {
            'dist':     np.zeros((arglist.max_episode_len, n, n)),
            'speed':    np.zeros((arglist.max_episode_len, n,)),
            'health':   np.zeros((arglist.max_episode_len, n,)),
            'fire':     np.zeros((arglist.max_episode_len, n,)),
            'bite':     np.zeros((arglist.max_episode_len, n, n)),
            'hit':      np.zeros((arglist.max_episode_len, n, n))
        }
        episode_info = {
            'dist':     np.zeros((arglist.num_episodes, n, n)),
            'speed':    np.zeros((arglist.num_episodes, n,)),
            'health':   np.zeros((arglist.num_episodes, n,)),
            'fire':     np.zeros((arglist.num_episodes, n,)),
            'bite':     np.zeros((arglist.num_episodes, n, n)),
            'hit':      np.zeros((arglist.num_episodes, n, n))
        }

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]

            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            # update episode step stats
            for key in step_info:
                step_info[key][episode_step] = info_n[key]

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            # record reward
            rewards[-1, :] += rew_n

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                rewards = np.concatenate((rewards, np.zeros((1, n))))

                # aggregate step_info
                episode_info['dist'][episode_number] = np.mean(step_info['dist'], axis=0)
                episode_info['speed'][episode_number] = np.mean(step_info['speed'], axis=0)
                episode_info['health'][episode_number] = np.min(step_info['health'], axis=0)
                episode_info['fire'][episode_number] = np.sum(step_info['fire'], axis=0)
                episode_info['bite'][episode_number] = np.sum(step_info['bite'], axis=0)
                episode_info['hit'][episode_number] = np.sum(step_info['hit'], axis=0)

                # reset step_info
                for key in step_info: step_info[key][:] = 0.

            # increment global step counter
            train_step += 1

            # for displaying policies while training
            if arglist.display and (episode_number % arglist.display_rate == 0) and episode_number > 0 and er_fill_frac_min >= 1.0:
                time.sleep(0.1)
                env.render()

            # update all trainers
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal:
                # check replay buffer status
                er_status = np.array([[len(t.replay_buffer), t.max_replay_buffer_len] for t in trainers])
                er_fill_frac = er_status[:, 0] / er_status[:, 1]
                er_fill_frac_min = er_fill_frac[np.argmin(er_fill_frac)]

                # print progress
                offset = -1 if train_step == 1 else -2
                print("steps: {}\tepisode: {}\treplay: {:.2f}%\treward: {}\ttime: {}".format(
                    train_step,
                    episode_number,
                    er_fill_frac_min * 100,
                    "\t".join(['[', *["%.2f" % r for r in list(rewards[offset])], ']']),
                    round(time.time()-t_start, 3))
                )

                t_start = time.time()

                # save state
                if (episode_number % arglist.save_rate == 0) and er_fill_frac_min >= 1.0:
                    print("saving...", end='')
                    # save policy snapshot
                    snapshot_folder = "{}/{}".format(arglist.save_dir, arglist.exp_name)
                    os.makedirs(snapshot_folder, exist_ok=True)
                    saver.save(U.get_session(), snapshot_folder + '/session', global_step=episode_number)
                    # save rewards
                    rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                    with open(rew_file_name, 'wb') as fp:
                        pickle.dump(rewards, fp)
                    # save stats
                    for key in episode_info:
                        stats_file_name = "{}{}_{}.pkl".format(arglist.plots_dir, arglist.exp_name, key)
                        with open(stats_file_name, 'wb') as fp:
                            pickle.dump(episode_info[key], fp)
                    print("done")

                episode_number += 1

            # saves final episode reward for plotting training curve later
            if episode_number == arglist.num_episodes:
                print('...Finished total of {} episodes.'.format(episode_number))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

