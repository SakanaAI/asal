# import taichi as ti
# import torch
# from eincasm_python.eincasm import Eincasm

# VIS_CHIDS = [('com', 'r'), ('com', 'g'), ('com', 'b')]

# ti.init(ti.gpu)

# @ti.dataclass
# class VisParams:
#     chindices: ti.types.vector(n=len(VIS_CHIDS), dtype=ti.i32)
#     chlims: ti.types.matrix(n=len(VIS_CHIDS), m=2, dtype=ti.f32)
#     scale: ti.i32
#     brush_radius: ti.i32
#     chnum_to_paint: ti.i32
#     chindex_to_paint: ti.i32
#     val_to_paint: ti.f32
#     val_to_paint_dt: ti.f32
#     drawing: ti.i32
#     mouse_posx: ti.f32
#     mouse_posy: ti.f32
#     perturb_strength: ti.f32
#     is_perturbing_weights: ti.i32
#     is_perturbing_biases: ti.i32
#     test: ti.f32

#     def __init__(self):
#         self.val_field=None

#     def mkfield(self,
#                 chindices: ti.types.vector(n=len(VIS_CHIDS), dtype=ti.i32),
#                 chlims: ti.types.matrix(n=len(VIS_CHIDS), m=2, dtype=ti.f32),
#                 scale: ti.i32,
#                 chindex_to_paint = None,
#                 brush_radius = 4,
#                 chnum_to_paint = 0,
#                 val_to_paint = 1.5,
#                 val_to_paint_dt = -1,
#                 drawing = False,
#                 mouse_posx = 0.0,
#                 mouse_posy = 0.0,
#                 perturb_strength = 0.1,
#                 is_perturbing_weights = False,
#                 is_perturbing_biases = False):
#         if chindex_to_paint is None:
#             chindex_to_paint = self.chindices[0]
#         val_field = VisParams.field(shape=())
#         val_field.chindices = chindices
#         val_field.chlims = chlims
#         val_field.scale = scale
#         val_field.chindex_to_paint = chindex_to_paint
#         val_field.brush_radius = brush_radius
#         val_field.chnum_to_paint = chnum_to_paint
#         val_field.val_to_paint = val_to_paint
#         val_field.val_to_paint_dt = val_to_paint_dt
#         val_field.drawing = drawing
#         val_field.mouse_posx = mouse_posx
#         val_field.mouse_posy = mouse_posy
#         val_field.perturb_strength = perturb_strength
#         val_field.is_perturbing_weights = is_perturbing_weights
#         val_field.is_perturbing_biases = is_perturbing_biases
#         self.val_field = val_field
#         return val_field
    
# ein = Eincasm(shape=(100, 100), torch_device=torch.device("mps"), num_com=5)
# world = ein.world
# e_chindices = world.get_inds_tivec(VIS_CHIDS)
# e_chlims = world.get_lims_timat(VIS_CHIDS)
# vpf = VisParams().mkfield(chindices=e_chindices, chlims=e_chlims, scale=1)
# @ti.kernel
# def tst(p: ti.template()):
#     pv = p[None]
#     pv.scale = 1
#     p[None] = pv


# print(vpf.scale)
# tst(vpf)
# print(vpf.scale)