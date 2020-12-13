import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # current number of zombie to human physical contacts
        self.biting = 0
        # health
        self.health = 1.0
        # fraction of time until can fire again (0: reloaded, 1: just fired)
        self.reloading = 0.0
        # which way is the agent aiming?
        self.aim_heading = None
        # angular velocity of aim
        self.aim_vel = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # arms action
        self.a = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.mass = 1.0

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot observe the world
        self.blind = False
        # can the agent shoot?
        self.armed = False
        # aim physics
        self.max_aim_vel = 2*np.pi
        self.arms_act_pres = 0.5
        self.arms_act_sens = 20
        self.arms_pallet_count = 6
        self.arms_pallet_damage = 0.05
        self.arms_pallet_range = 10
        self.arms_pallet_spread = 10 /360.0*2*np.pi
        # time taken to reload
        self.arms_reload_time = 0.5
        # physical motor noise amount
        self.u_noise = None
        # arms handling noise
        self.a_noise = None
        # control range
        self.u_range = 1.0
        # team membership
        self.team = 0
        # state
        self.state = AgentState()
        self.health_decay = 1.0
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # projectile paths
        self.projectiles = None
        # info
        self.info = None

    # return all entities in the world
    @property
    def entities(self):
        return self.agents

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        n = len(self.agents)
        for i, agent in enumerate(self.agents): agent.index = i
        self.info = {
            'dist': np.zeros((n, n)),
            'speed': np.zeros((n,)),
            'health': np.zeros((n,)),
            'fire': np.zeros((n,)),
            'bite': np.zeros((n, n)),
            'hit': np.zeros((n, n))
        }
        # record distance
        for agent in self.agents:
            for other in self.agents:
                _, self.info['dist'][agent.index, other.index] = self.distance(agent, other)
                
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        self.projectiles = np.zeros((0, 4))
        for agent in self.agents:
            self.update_agent_state(agent)

        # record health
        for agent in self.agents:
            self.info['health'][agent.index] = agent.state.health

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        # wall forces
        for i, entity in enumerate(self.entities):
            walls = np.array([
                [[0, 1], [0, -1]], # top
                [[-1, 0], [1, 0]], # left
                [[0, -1], [0, 1]], # bottom
                [[1, 0], [-1, 0]], # right
            ])
            for j in range(walls.shape[0]):
                f = self.get_wall_force(entity, walls[j])
                if f is not None:
                    if p_force[i] is None: p_force[i] = 0.0
                    p_force[i] += f
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                max_speed = entity.max_speed * entity.state.health
                if speed > max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

            # record speed
            self.info['speed'][entity.index] = np.sqrt(np.sum(np.square(entity.state.p_vel)))

    def update_agent_state(self, agent):

        # compute ballistics
        if agent.armed:

            # aim direction
            noise = np.random.randn(*agent.action.a.shape) * agent.a_noise if agent.a_noise else 0.0
            agent.state.aim_vel = (agent.action.a[0] + noise) * agent.max_aim_vel
            agent.state.aim_heading += agent.state.aim_vel * self.dt
            agent.state.aim_heading %= 2 * np.pi

            # firing
            if agent.state.reloading > 0:
                reload_amount = self.dt / agent.arms_reload_time
                agent.state.reloading = np.clip(agent.state.reloading - reload_amount, 0.0, 1.0)
            else:
                a_force = agent.action.a[1] + noise
                act_prob = 1/(1+np.exp(agent.arms_act_sens * (agent.arms_act_pres - a_force)))
                activated = np.random.binomial(1, act_prob * agent.state.health)
                if activated:
                    # record fire
                    self.info['fire'][agent.index] += 1

                    agent.state.reloading = 1.0

                    # create rays representing projectiles
                    ray_pos = agent.state.p_pos + np.array([np.cos(agent.state.aim_heading), np.sin(agent.state.aim_heading)]) * agent.size * 1.5
                    ray_ang = np.random.normal(agent.state.aim_heading, agent.arms_pallet_spread, agent.arms_pallet_count)
                    rays = np.array([np.cos(ray_ang), np.sin(ray_ang)]).transpose()
                    others = np.array([other for other in self.agents if other is not agent])
                    dists = np.full((rays.shape[0], len(others)), np.inf)

                    # find intersecting points with other agents
                    for o, other in enumerate(others):
                        # algo adapted from ray-sphere collision in github.com/adamlwgriffiths/Pyrr
                        delta_pos = ray_pos - other.state.p_pos
                        b = 2 * np.dot(rays, delta_pos)
                        c = np.dot(delta_pos, delta_pos) - other.size ** 2
                        delta = b ** 2 - 4 * c
                        hits = np.arange(delta.shape[0])[delta >= 0]
                        q = -0.5 * (b[hits] + ((b[hits] > 0) * 2 - 1) * np.sqrt(delta[hits]))
                        d = np.min(np.array([q, c/q]), axis=0)
                        dists[hits, o] = d

                    # figure out which agents were hit
                    dists[dists < 0] = np.inf
                    closest = np.argmin(dists, axis=1)
                    min_dists = np.min(dists, axis=1)
                    closest_in_range = closest[min_dists < agent.arms_pallet_range]
                    min_dists = np.clip(min_dists, 0, agent.arms_pallet_range)

                    # deduct health from agents that were hit
                    for other in others[closest_in_range]:
                        other.state.health = np.clip(other.state.health - agent.arms_pallet_damage, 0.0, 1.0)
                        # record hit
                        self.info['hit'][agent.index, other.index] += 1

                    # ray segments for rendering
                    new_projectiles = np.concatenate((
                        np.tile(ray_pos, (rays.shape[0], 1)),
                        ray_pos + rays * min_dists.reshape((-1, 1))
                    ), axis=1)
                    self.projectiles = np.concatenate((self.projectiles, new_projectiles), axis=0)

        # effects of human-zombie collisions
        for agent in self.agents:
            agent.state.biting = 0
        for a, agent_a in enumerate(self.agents):
            for b, agent_b in enumerate(self.agents):
                if(b <= a): continue
                # skip if same team
                if(agent_a.team == agent_b.team): continue
                # skip if not touching
                _, dist = self.distance(agent_a, agent_b)
                min_dist = agent_a.size + agent_b.size
                if(dist > min_dist): continue
                # increase biting count
                agent_a.state.biting += 1
                agent_b.state.biting += 1
                human, zombie = (agent_a, agent_b) if agent_a.team == 0 else (agent_b, agent_a)
                human.state.health *= human.health_decay

                # record bite
                self.info['bite'][zombie.index][human.index] += 1

    def distance(self, agent, other):
        delta_pos = agent.state.p_pos - other.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return (delta_pos, dist)

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos, dist = self.distance(entity_a, entity_b)
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    def get_wall_force(self, entity, wall):
        if (not entity.collide) or (not entity.movable): return None
        wall_pos, wall_norm = wall[0], wall[1]
        dist = (entity.state.p_pos - wall_pos) @ wall_norm
        dist_min = entity.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * penetration * wall_norm
        return force

