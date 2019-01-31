import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from copy import deepcopy

class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, reward_function=None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        self.x_threshold = 2.4
        self.world_width = 2.0 * self.x_threshold

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.world_width,
            np.finfo(np.float32).max,
            math.pi,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(np.array([-1.0]), np.array([1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        if reward_function is None:
            def reward(cart_pole):
                if cart_pole.state[0] < -self.x_threshold or cart_pole.state[0] > self.x_threshold:
                  return -1
                return 1 if -0.1 <= angle_normalize(cart_pole.state[2]) <= 0.1 else 0
            self.reward = reward
        else:
            self.reward = reward_function

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * action[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x,x_dot,theta,theta_dot)

        done =  x < -self.x_threshold or x > self.x_threshold
        done = bool(done)
        
        reward = self.reward(self)

        return np.expand_dims(self.state, axis=0), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[2] += np.pi
        self.steps_beyond_done = None
        return np.expand_dims(self.state, axis=0)

    def render(self, mode='human', goal_state=None):
        x_org = 100
        track_width = 600
        screen_width = track_width + 2 * x_org
        screen_height = 400

        scale = track_width/self.world_width
        carty = 150 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        arrowwidth = 7.0
        arrowlenmax = 2*cartwidth
        max_cart_vel = 10.0 # TODO: deal with this hardcode
        max_pole_vel = 10.0 # TODO: deal with this hardcode

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Track
            self.track = rendering.Line((x_org,carty), (x_org+track_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            # Cart
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            #axleoffset =cartheight/4.0
            axleoffset = 0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            # Pole
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform()
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            # Axle
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            

            # render goal state from HL agent
            if goal_state is not None:
                self.cart_goal = deepcopy(cart)
                #cart_goal_frame = rendering.make_polygon(self.cart_goal.v,filled=False)
                
                self.pole_goal = deepcopy(pole)
                #pole_goal_frame = rendering.make_polygon(self.pole_goal.v,filled=False)

                self.cart_goal.set_color(.7,.3,.7) #pink?
                self.pole_goal.set_color(.8,.2,.8) #pinker?

                self.viewer.add_geom(self.cart_goal)
                self.viewer.add_geom(self.pole_goal)
                #self.viewer.add_geom(cart_goal_frame)
                #self.viewer.add_geom(pole_goal_frame)

                self.carttrans_goal = self.cart_goal.attrs[-1]
                self.poletrans_goal = self.pole_goal.attrs[-2]
                self.pole_goal.attrs[-1] = self.carttrans_goal # so that we can just change this once below
                
                #cart_goal_frame.add_attr(self.carttrans_goal)
                #pole_goal_frame.add_attr(self.poletrans_goal)
                #pole_goal_frame.add_attr(self.carttrans_goal)


                # Arrow points
                l,r,t,b, m = 0, arrowlenmax, arrowwidth/2, -arrowwidth/2, 0

                # Cart velocity goal ARROW
                goal_cart_arrow_body = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                goal_cart_arrow_head = rendering.FilledPolygon([(r,3*t), (r+4*t,m), (r,3*b)])
                
                ac = polelen
                goal_pole_arrow_body = rendering.FilledPolygon([(l,b+ac), (l,t+ac), (r,t+ac), (r,b+ac)])
                goal_pole_arrow_head = rendering.FilledPolygon([(r,3*t+ac), (r+4*t,m+ac), (r,3*b+ac)])

                #arrow_color = [0, .45, .45] #green
                arrow_color = [.45, .45, .45] #green
                goal_cart_arrow_body.set_color(*arrow_color) 
                goal_cart_arrow_head.set_color(*arrow_color) 
                goal_pole_arrow_body.set_color(*arrow_color) 
                goal_pole_arrow_head.set_color(*arrow_color) 

                self.goal_cart_arrow_trans = rendering.Transform()
                goal_cart_arrow_body.add_attr(self.goal_cart_arrow_trans)
                goal_cart_arrow_head.add_attr(self.goal_cart_arrow_trans)
                self.goal_pole_arrow_trans = rendering.Transform()
                goal_pole_arrow_body.add_attr(self.goal_pole_arrow_trans)
                goal_pole_arrow_head.add_attr(self.goal_pole_arrow_trans)
                self.viewer.add_geom(goal_cart_arrow_body)
                self.viewer.add_geom(goal_cart_arrow_head)
                self.viewer.add_geom(goal_pole_arrow_body)
                self.viewer.add_geom(goal_pole_arrow_head)


            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        # render goal state from HL agent
        if goal_state is not None:
            cartx_goal = goal_state[0]*scale+screen_width/2.0 # MIDDLE OF CART
            self.carttrans_goal.set_translation(cartx_goal, carty)
            self.poletrans_goal.set_rotation(-goal_state[2]) # TODO

            # Cart velocity goal ARROW
            self.goal_cart_arrow_trans.set_translation(cartx_goal, carty)
            cart_arrow_scale = goal_state[1] / max_cart_vel
            self.goal_cart_arrow_trans.set_scale(newy=1, newx=cart_arrow_scale)

            # Pole velocity goal ARROW
            self.goal_pole_arrow_trans.set_translation(cartx_goal, carty)
            pole_arrow_scale = goal_state[3] / max_pole_vel
            self.goal_pole_arrow_trans.set_scale(newy=1, newx=pole_arrow_scale)
            self.goal_pole_arrow_trans.set_rotation(-goal_state[2])
            
            # Goal cart color
            #goal_cart_vel_norm = goal_state[1] / max_cart_vel
            #(red, blue) = (1, 0) if goal_cart_vel_norm > 0 else (0,1)
            #self.cart_goal.set_color(red, 0, blue, abs(goal_cart_vel_norm))

            # Goal pole color
            #goal_pole_vel_norm = goal_state[3] / max_pole_vel
            #(red, blue) = (1, 0) if goal_pole_vel_norm > 0 else (0,1)
            #self.pole_goal.set_color(red, 0, blue, abs(goal_pole_vel_norm))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
