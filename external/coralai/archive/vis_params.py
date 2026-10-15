import taichi as ti

VIS_CHIDS = [('com', 'r'), ('com', 'g'), ('com', 'b')]

@ti.dataclass
class VisParams:
    chindices: ti.types.vector(n=len(VIS_CHIDS), dtype=ti.i32)
    chlims: ti.types.matrix(n=len(VIS_CHIDS), m=2, dtype=ti.f32)
    scale: ti.i32
    brush_radius: ti.i32
    chnum_to_paint: ti.i32
    chindex_to_paint: ti.i32
    val_to_paint: ti.f32
    val_to_paint_dt: ti.f32
    drawing: ti.i32
    mouse_posx: ti.f32
    mouse_posy: ti.f32
    perturb_strength: ti.f32
    is_perturbing_weights: ti.i32
    is_perturbing_biases: ti.i32
    test: ti.f32

    def __init__(self,
                 chindices: ti.types.vector(n=len(VIS_CHIDS), dtype=ti.i32),
                 chlims: ti.types.matrix(n=len(VIS_CHIDS), m=2, dtype=ti.f32),
                 scale: ti.i32,
                 chindex_to_paint = None,
                 brush_radius = 4,
                 chnum_to_paint = 0,
                 val_to_paint = 1.5,
                 val_to_paint_dt = -1,
                 drawing = False,
                 mouse_posx = 0.0,
                 mouse_posy = 0.0,
                 perturb_strength = 0.1,
                 is_perturbing_weights = False,
                 is_perturbing_biases = False,
                 test = -1):
        if chindex_to_paint is None:
            chindex_to_paint = self.chindices[0]
        self.chindices = chindices
        self.chlims = chlims
        self.scale = scale
        self.chindex_to_paint = chindex_to_paint
        self.brush_radius = brush_radius
        self.chnum_to_paint = chnum_to_paint
        self.val_to_paint = val_to_paint
        self.val_to_paint_dt = val_to_paint_dt
        self.drawing = drawing
        self.mouse_posx = mouse_posx
        self.mouse_posy = mouse_posy
        self.perturb_strength = perturb_strength
        self.is_perturbing_weights = is_perturbing_weights
        self.is_perturbing_biases = is_perturbing_biases
        self.test = test
        
        # struct_type = ti.types.struct(**self.type_dict)
        # val_field = struct_type.field(shape=())
        # for k, v in self.val_dict.items():
        #     val_field[None][k] = v
        # return val_field, struct_type