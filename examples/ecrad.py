from gt4py.next.ffront.fbuiltins import maximum, minimum, astype
from gt4py.next.ffront.experimental import as_offset
from gt4py import next as gtx

dtype = float

Column = gtx.Dimension("Column")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
GPoint = gtx.Dimension("XGPoint",kind=gtx.DimensionKind.LOCAL)
ER = gtx.Dimension("YER", kind=gtx.DimensionKind.LOCAL)

GPointRE = gtx.Field[gtx.Dims[GPoint, ER], dtype]
ColumnK = gtx.Field[gtx.Dims[Column, K], dtype]
ColumnKGPoint = gtx.Field[gtx.Dims[Column, K, GPoint], dtype]

_DIM_ORDER = {Column: 0, K: 1, GPoint: 2, ER: 3}
def _ordered_dims(dims: list[gtx.Dimension]) -> list[gtx.Dimension]:
    return sorted(dims, key=lambda dim: _DIM_ORDER[dim])

gtx.common._ordered_dims = _ordered_dims

def f_add_optical_properties(
    asymmetry: GPointRE,
    cloud_fraction: ColumnK,
    effective_radius: ColumnK,
    mass_ext: GPointRE,
    ssa: GPointRE,
    water_path: ColumnK,
    od: ColumnKGPoint,
    scat_od: ColumnKGPoint,
    scat_asymmetry: ColumnKGPoint,
    d_effective_radius: dtype,
    effective_radius_0: dtype,
    DO_SCATTERING: bool,
    N_EFFECTIVE_RADII: int,
):
    if DO_SCATTERING:
        # compute absorption and scattering properties
        if cloud_fraction > 0.0:
            re_index = maximum(
                1.0,
                minimum(
                    1.0 + (effective_radius - effective_radius_0) / d_effective_radius,
                    N_EFFECTIVE_RADII - 0.0001,
                ),
            )
            ire = astype(re_index, int)
            weight2 = re_index - ire
            weight1 = 1.0 - weight2

            od_local = water_path * (
                    weight1 * mass_ext[ER(ire-1)]+ weight2 * mass_ext[ER(ire)]
                )
            od += od_local
            od_local *= weight1 * ssa[ER(ire - 1)] + weight2 * ssa[ER(ire)]
            scat_od += od_local
            scat_asymmetry += od_local * (
                weight1 * asymmetry[ER(ire - 1)] + weight2 * asymmetry[ER(ire)]
            )
            return od, scat_od, scat_asymmetry
            
                
    # else:
    #     # compute absorption properties only
    #     if water_path > 0.0:
    #         re_index = max(
    #             1.0,
    #             min(
    #                 1.0 + (effective_radius - effective_radius_0) / d_effective_radius,
    #                 N_EFFECTIVE_RADII - 0.0001,
    #             ),
    #         )
    #         ire = int(re_index)
    #         weight2 = re_index - ire
    #         weight1 = 1.0 - weight2
    #         for n_gpoint in range(0, N_GPOINTS):
    #             od[0, 0, 0][n_gpoint] += (
    #                 water_path
    #                 * (weight1 * mass_ext.A[n_gpoint, ire - 1] + weight2 * mass_ext.A[n_gpoint, ire])
    #                 * (1.0 - (weight1 * ssa.A[n_gpoint, ire - 1] + weight2 * ssa.A[n_gpoint, ire]))
    #             )


GPointRE = gtx.Field[gtx.Dims[GPoint, ER], dtype]
ColumnK = gtx.Field[gtx.Dims[Column, K], dtype]
ColumnKGPoint = gtx.Field[gtx.Dims[Column, K, GPoint], dtype]

n_gpoint = 32
n_er = 50
n_cols = 1
n_k = 137

gpoint_re_domain = gtx.domain({GPoint: n_gpoint, ER: n_er})
column_k_domain = gtx.domain({Column: n_cols, K: n_k})
column_k_gpoint_domain = gtx.domain({Column: n_cols, K: n_k, GPoint: n_gpoint})

asymmetry = gtx.zeros(gpoint_re_domain, dtype=dtype)
cloud_fraction = gtx.zeros(column_k_domain, dtype=dtype)
effective_radius = gtx.zeros(column_k_domain, dtype=dtype)
mass_ext = gtx.zeros(gpoint_re_domain, dtype=dtype)
ssa = gtx.zeros(gpoint_re_domain, dtype=dtype)
water_path = gtx.zeros(column_k_domain, dtype=dtype)
od = gtx.zeros(column_k_gpoint_domain, dtype=dtype)
scat_asymmetry = gtx.zeros(column_k_gpoint_domain, dtype=dtype)
scat_od = gtx.zeros(column_k_gpoint_domain, dtype=dtype)

od = f_add_optical_properties(
    asymmetry=asymmetry,
    cloud_fraction=cloud_fraction,
    effective_radius=effective_radius,
    mass_ext=mass_ext,
    ssa=ssa,
    water_path=water_path,
    od=od,
    scat_asymmetry=scat_asymmetry,
    scat_od=scat_od,
    d_effective_radius=10.0,
    effective_radius_0=10.0,
    DO_SCATTERING=True,
    N_EFFECTIVE_RADII=n_er)
