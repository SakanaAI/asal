import numpy as np
from env import CAEnvironment as caenv
import utils.bcolors as bcolors
import math

'''
env = caenv.CAEnvironment("slime_eg")
env.gen_padded_food()
env.innoculate()
env.display()

ag = caag.CAAgent("Lame Agent")
def lame_rules(chunk, env):
    return [1] * env.n_channels

ag.set_rule_func(lame_rules)

rag = caag.CAAgent("Random Agent")
def rand_rules(chunk, env):
    return np.random.random(env.n_channels)
rag.set_rule_func(rand_rules)
print(ag.display())
print(rag.display())


env.start_new_video(channels=(env.food_i, env.life_i), cmaps = (cm.copper,cm.gray))
for _ in range(30):
    # ag.apply_to_env(env, log=True, vid_speed=3)
    rag.apply_to_env(env, log=True, vid_speed=5)
env.display()
Video(env.save_video())
'''


class CAAgent:
    
    def __init__(self, id: str, rule_func: callable, kernel: tuple = (3, 3), input_channels: tuple=(0),
                output_channels: tuple=(0)):
        """Initializes a CAAgent with the specified ID, rule function, kernel, and input/output channels.

        Parameters:
            id (str): The ID of the agent
            rule_func (callable): The rule function of the agent
                - input: chunk of environment of size (kernel[0], kernel[1], len(input_channels))
                - output: the resultant center cell channels of size len(output_channels)
            kernel (tuple): The size of the kernel (w,h), moore neighborhoods are used
            input_channels (tuple): The indices of environmental channels to use as input, in order
            output_channels (tuple): The indices of environmental channels to output to, in order
        """
        self.id = id
        self.kernel = kernel
        self.rule_func = rule_func
        self.input_channels = input_channels
        self.output_channels = output_channels

    def display(self):
        return ("CAAgent ID: {0}" +
                "\n\tKERNEL: {1}".format(self.id, self.kernel))

    def _apply_once(self, env: caenv.CAEnvironment, i: int, j: int):
        # Applies the rule function to the cell at (i,j) in the environment.
        input = env.channels[:, self.kernel_full[0] + i,
                             self.kernel_full[1] + j]
        if input[env.life_i].sum() > 0:
            desires = self.apply_rules(input.flatten(), env)
            env.update_chunk(i, j, desires)           

    # Stochastically applies agent to every alive cell and its neighbors, once
    # - think of this as the CA releasing DNA into its neigbors but not necessarily
    #   forming a cell at that point
    def apply_to_env(self, env: caenv.CAEnvironment, dropout: float = 0.25):
        if self.apply_rules is None:
            print(bcolors.WARNING +
                  "ca_agent.py:apply_to_env: Must set rule function before applying to an environment" + bcolors.ENDC)
            return

        if dropout <= 1:
            # N random samples from 0 to width*height, where N = width*height*(1-dropout)
            inds = np.random.choice(
                (env.width)*(env.height), (int)((env.width)*(env.height)*(1-dropout)))
        else:
            inds = np.arange(0, env.width*env.height)

        # Convert indices to (x,y) coordinates
        coords = np.unravel_index(inds, (env.width, env.height))

        padding = (self.kernel[0]-1)//2
        
        """TODO: Parallelize this, it's the bottleneck. Below are instructions for doing so.
        
        
        """
        for i, j in zip(coords[0]+1, coords[1]+1):
            


        # total_steps = 0
        # for i, j in zip(coords[0]+1, coords[1]+1):
        #     input = env.channels[:, self.kernel_full[0] + i,
        #                          self.kernel_full[1] + j]
        #     if input[env.life_i].sum() > 0:
        #         desires = self.apply_rules(input.flatten(), env)
        #         env.update_chunk(i, j, desires)

        #         if log and vid_speed < 10 and total_steps % (math.pow(2, vid_speed)) == 0:
        #             env.add_state_to_video()

        #         total_steps += 1
        # if log:
        #     env.add_state_to_video()
