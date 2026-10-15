import taichi as ti

ti.init(arch=ti.gpu)

flow_kernel = ti.Vector.field(2, dtype=ti.i32, shape=5)
flow_kernel[0] = [0, 0]  # ORIGIN
flow_kernel[1] = [-1, 0] # UP
flow_kernel[2] = [0, 1]  # RIGHT
flow_kernel[3] = [1, 0]  # DOWN
flow_kernel[4] = [0, -1] # LEFT

sense_kernel = ti.Vector.field(2, dtype=ti.i32, shape=5)
sense_kernel[0] = [0, 0]  # ORIGIN
sense_kernel[1] = [-1, 0] # UP
sense_kernel[2] = [0, 1]  # RIGHT
sense_kernel[3] = [1, 0]  # DOWN
sense_kernel[4] = [0, -1] # LEFT

num_com = 3
ch_types = {
    'capital': ti.f32,
    'obstacle': ti.f32,
    'muscles': ti.types.vector(flow_kernel.shape[0], dtype=ti.f32),
    'com': ti.types.vector(num_com, dtype=ti.f32),
    'fact': ti.f32,
    'growth_act': ti.types.vector(flow_kernel.shape[0], dtype=ti.f32),
}
celltype = ti.types.struct(**ch_types)
world = celltype.field(shape=(10,10))

@ti.func
def ReLu(x):
    return x if x > 0 else 0

senses_neigh = {
    'capital': False, # senses neighborhood?
    'muscles': False,
    'com': True,
    'obstacle': True
}
sensors = senses_neigh.keys()
actuators = ['fact', 'growth_act']
num_genomes = 3
sensor_sizes = [
    {'capital': 1, 'muscles': 2, 'com': 3, 'obstacle': 5},
    {'capital': 6, 'muscles': 7, 'com': 1, 'obstacle': 3},
    {'capital': 1, 'muscles': 1, 'com': 6, 'obstacle': 4},
]

latent_sizes = [10,2,3]
actuator_sizes = [
    {'fact': 1, 'growth_act': 2},
    {'fact': 3, 'growth_act': 4},
    {'fact': 5, 'growth_act': 6},
]

activation_funcs = [ti.tanh, ReLu, ti.sin]

def gen_sensor_weights(sensors, ch_types, sense_kernel, sensor_sizes):
    all_sensor_weights = []
    for genome_id in range(len(sensor_sizes)): # num_genomes
        w = {}
        for sensor, senses_neigh in sensors.items():
            if senses_neigh:
                w[sensor] = ti.field(ch_types[sensor], shape=(sense_kernel.shape[0], sensor_sizes[genome_id][sensor]))
            else:
                w[sensor] = ti.field(ch_types[sensor], shape=(sensor_sizes[genome_id][sensor]))
        all_sensor_weights.append(w)
    return all_sensor_weights

sensor_weights = gen_sensor_weights(sensors, ch_types, sense_kernel, sensor_sizes)

sensor_layers = []
for genome_ss in sensor_sizes:
    layers = {}
    for sensor, layer_size in genome_ss.items():
        layers[sensor] = ti.types.vector(n=layer_size, dtype=ti.f32)
    sensor_layers.append(ti.types.struct(**layers))

print(f"Sensor Layers: {sensor_layers}\n\nSensor Weights: {sensor_weights}")

# @ti.kernel
# def forward():
    
        
# def gen_actuator_weights(actuators, ch_types, actuator_sizes):
#     all_actuator_weights = []
#     for genome_id in range(len(actuator_sizes)): # num genomes
#         w = {}
#         for actuator in actuators:
#             w[actuator] = ti.field(ch_types[actuator], shape=(latent_sizes[genome_id], actuator_sizes[genome_id][actuator]))
#         all_actuator_weights.append(w)
#     return all_actuator_weights

# def gen_latent_weights(sensor_sizes, latent_sizes, actuator_sizes):
#     incoming_latent_weights = []
#     outgoing_latent_weights = []
#     for genome_id in range(num_genomes):
#         sensor_latent_w = {}
#         for sensor in sensors.keys():
#             sensor_latent_w[sensor] = ti.field(ti.f32, shape=(sensor_sizes[genome_id][sensor], latent_sizes[genome_id]))

#         latent_actuator_w = {}
