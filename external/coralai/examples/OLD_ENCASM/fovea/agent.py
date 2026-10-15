import numpy as np
import ca_environment as caenv
import math
from bcolors import bcolors


class CAAgent:

    # For slime mold/traditional agenst
    apply_rules = None

    # For random walk agents
    apply_walk = None
    foveal_size = 1  # arbitrary for now
    n_spatial_chs = 2  # arbitrary for now

    # apply_rules must be in the form f()

    def __init__(self, id, kernel="von_n"):
        self.id = id
        self.kernelid = kernel
        # ----- Neighborhood/Channel Parameters -----
        moore = (np.array([1,  1,  1,  0,  0,  -1, -1, -1]),
                 np.array([-1,  0,  1, -1,  1,  -1,  0,  1]))  # Moore neigh indices

        von_n = (np.array([-1,  0,  0, 1]),
                 np.array([0, -1,  1,  0]))  # Von Neumann neighborhood indices

        moore_f = (np.array([1,  1,  1,  0,  0,  0,  -1, -1, -1]),
                   np.array([-1,  0,  1, -1,  0,  1,  -1,  0,  1]))  # Includes center

        von_n_f = (np.array([-1,  0,  0,  0, 1]),
                   np.array([0, -1,  0,  1,  0]))  # Includes center

        if kernel == "moore":
            self.kernel = moore
            self.kernel_full = moore_f  # Incliudes center
        else:
            self.kernel = von_n
            self.kernel_full = von_n_f  # Incliudes center

        self.n_neighs = len(self.kernel[0])

    def display(self):
        return ("CAAgent ID: {0}" +
                "\n\tKERNEL: {1}" +
                "\n\tAGENT_TYPE: {2}" +
                "{3}").format(self.id, self.kernelid,
                              'foveal_walk' if self.apply_walk is not None else 'slime mold',
                              "\n\t\tFOVEAL_CHs: {0}\n\t\tSPATIAL_CHs: {1}".format(self.foveal_size, self.n_spatial_chs) if self.apply_walk is not None else "")
        # (f"CAAgent, ID: {self.id}" +
        #  "\n\tAGENT TYPE: " +
        #  f"{'foveal walk' if self.apply_rules is None else('slime_mold' if self.apply_walk is None else 'no ruleset')}" +
        #  f"{f'\n\t\tfoveal_size: {self.foveal_size}\n\t\tn_spatial_channels: {self.n_spatial_chs}' if self.apply_walk is not None else ''}"
        #  "\n\tKERNEL: {self.kernelid}\n\t")

    def set_rule_func(self, func):
        self.apply_rules = func

    def set_walk_func(self, func):
        self.apply_walk = func

    def n_walk_inputs(self):
        return len(self.kernel_full[0]) * (self.n_spatial_chs + 1) + self.foveal_size

    def n_walk_outputs(self):
        return 2 + self.n_spatial_chs + self.foveal_size

    def apply_walk_to_env(self, env: caenv.CAEnvironment, stride=None, max_steps=None, max_dist=None, step_penalty=None, log=False, vid_speed=10):
        '''
        A walk takes as input:
            A "foveal" set of memory channels (output of the NN)
            The current neighborhood, including any food cells and hidden channels stored by the agent
        And produces as ouptut:
            A horizontal vector for where to go next
            A vertical vector for where to go next
            The new foveal memory, passed to the next iteration
            A spatial memory to be stored in the center of the neighborhood
        The quality of a walk is judged by how much food it gathers in a set number of steps (or a set total distance travelled)
        Step penalty is a constant distance added to each step, so that taking many steps is more costly than one large one
            It should scale with the size of the environment and max_dist (it doesn't affect max_steps)
        '''
        if stride is None:
            stride = env.esize/2
        if max_dist is not None and step_penalty is None:
            step_penalty = (max_dist/env.esize) * 0.1
        if self.n_spatial_chs > env.n_hidden:
            print(bcolors.FAIL + "ca_agent.py:apply_walk_to_env: Must apply to an environment with n_hidden channels == agents n_spatial channels" + bcolors.ENDC)
        if self.apply_walk is None:
            print(bcolors.FAIL +
                  "ca_agent.py:apply_walk_to_env: Must set walk function before applying to an environment" + bcolors.ENDC)
            return
        if (max_steps is not None and max_dist is not None) or (max_steps is None and max_dist is None):
            print(bcolors.FAIL +
                  "ca_agent.py:apply_walk_to_env: Must specify either n_steps or tot_distance " + bcolors.ENDC)
            return

        coords = [np.random.randint(
            env.pad, env.cutsize), np.random.randint(env.pad, env.cutsize)]
        tot_steps = 0
        tot_dist = 0
        running = True
        foveal_mem = np.random.random(self.foveal_size)
        coords_hist = np.array([coords])
        food_count = 0

        while running:
            within_env = (coords[0] > env.pad-1 and coords[0] < env.eshape[1] and
                          coords[1] > env.pad-1 and coords[1] < env.eshape[1])
            if not within_env:
                food_count -= 1
                input = np.zeros(env.n_channels * len(self.kernel_full[0]))
            else:
                food_count += env.channels[env.food_i, coords[0], coords[1]]
                env.channels[env.food_i, coords[0], coords[1]] = 0

                input = (env.channels[:, self.kernel_full[0] + coords[0],
                                      self.kernel_full[1] + coords[1]]).flatten()

            output = self.apply_walk(
                input, foveal_mem, self.n_spatial_chs, env)
            if within_env:
                env.channels[env.hidden_i:, coords[0], coords[1]
                             ] = output[2+self.foveal_size:]
                print("out ", output[2+self.foveal_size:])
            dx = output[0] * stride
            coords[0] = int(coords[0] + dx)
            dy = output[1] * stride
            coords[1] = int(coords[1] + dy)

            # coords[1] = int(min(env.eshape[2]-env.pad-2,
            #                     max(coords[1] + dy, env.pad)))

            # print(dx, dy, "\t", output[0], output[1], env.eshape)

            foveal_mem = np.array(output[2:self.foveal_size+2])

            if log:
                coords_hist = np.append(coords_hist, [coords], axis=0)

            if log and vid_speed < 10 and tot_steps % (math.pow(2, vid_speed)) == 0:
                env.add_state_to_video()

            if max_dist is not None:
                tot_dist += dx + dy + step_penalty
                running = tot_dist < max_dist
            elif max_steps is not None:
                tot_steps += 1
                running = tot_steps < max_steps
        if log:
            env.add_state_to_video()
            return food_count, coords_hist

        return food_count
