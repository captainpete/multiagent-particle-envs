import numpy as np
from multiagent.core import World, Agent
from multiagent.scenario import BaseScenario

# 2D simple physics with agent collissions bordered by a force field.

# action space:
#   2D continuous action space (up/down, left/right)

# observation space:
#   Complete information on 2D positions of all agents (continuous 4D *
#   agents). This includes absolute position and velocity for the agent itself,
#   and relative position and velociry for other humans, and other zombies
#   (agents have identity by index position).

# reward function:
#   When a zombie and human come into contact then 10 reward is deducted from
#   the human and awarded to the zombie for every simulation step they remain
#   in contact.

class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 2
        num_humans = 2
        num_zombies = 5
        num_agents = num_humans + num_zombies

        # add agents
        agents = []
        # add humans' team
        for i in range(num_humans):
            agent = Agent()
            agent.collide = True
            agent.silent = False
            agent.team = 0
            agent.name = 'human %d' % i
            agent.size = 0.05
            agent.accel = 4.0
            agent.max_speed = 1.3
            agents.append(agent)
        # add zombies' team
        for i in range(num_zombies):
            agent = Agent()
            agent.collide = True
            agent.silent = False
            agent.team = 1
            agent.name = 'zombie %d' % i
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 1.0
            agents.append(agent)
        world.agents = agents

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        # random properties for agents
        team_colors = [
            np.array([0.35, 0.85, 0.35]),
            np.array([0.85, 0.35, 0.35])
        ]
        for i, agent in enumerate(world.agents):
            agent.color = np.array(team_colors[agent.team])
        # set random initial states
        for agent in world.agents:
            if agent.team == 0:
                agent.state.p_pos = np.random.uniform(-.9, .9, world.dim_p)
            elif agent.team == 1:
                agent.state.p_pos = np.random.uniform(-.9, .9, world.dim_p)
            else:
                raise f"don't know where to start agent in team {agent.team}"

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def distance(self, agent, other):
        delta_pos = agent.state.p_pos - other.state.p_pos
        return np.sqrt(np.sum(np.square(delta_pos)))

    def is_collision(self, agent1, agent2):
        dist = self.distance(agent1, agent2)
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def humans(self, world):
        return [agent for agent in world.agents if agent.team == 0]

    def zombies(self, world):
        return [agent for agent in world.agents if agent.team == 1]

    def reward(self, agent, world):
        if agent.team == 0:
            return self.human_reward(agent, world)
        elif agent.team == 1:
            return self.zombie_reward(agent, world)
        else:
            raise "Unknown reward for team %d" % agent.team

    def human_reward(self, agent, world):
        rew = 0
        zombies = self.zombies(world)

        # remove reward for being attacked by a zombie
        if agent.collide:
            for a in zombies:
                if self.is_collision(a, agent):
                    rew -= 10

        return rew

    def zombie_reward(self, agent, world):
        rew = 0
        humans = self.humans(world)

        # add reward for attacking a human
        for a in humans:
            if self.is_collision(a, agent):
                rew += 10

        return rew

    def human_observation(self, agent, world):
        other_pos = []
        other_vel = []

        humans = self.humans(world)
        for other in humans:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel - agent.state.p_vel)

        zombies = self.zombies(world)
        for other in zombies:
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel - agent.state.p_vel)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel)

    def zombie_observation(self, agent, world):
        other_pos = []
        other_vel = []

        humans = self.humans(world)
        for other in humans:
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel - agent.state.p_vel)

        zombies = self.zombies(world)
        for other in zombies:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel - agent.state.p_vel)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel)

    def observation(self, agent, world):
        if agent.team == 0:
            return self.human_observation(agent, world)
        elif agent.team == 1:
            return self.zombie_observation(agent, world)
        else:
            raise "Unknown observation for team %d" % agent.team

