"""CPU fallback for mmdet3d's numba.cuda rotated-IoU kernel.

WHY. mmdet3d's KITTI evaluator computes BEV/3D IoU with
`rotate_iou_gpu_eval`, a numba.cuda kernel. Whether that kernel compiles is a
property of the HOST, not of the code:

  * a host with a complete CUDA toolkit (libnvvm and libdevice from the same
    /usr/local/cuda) compiles it fine.
  * a host that mixes components -- e.g. the system's CUDA-12 libnvvm with the
    Debian CUDA-11.5 toolkit's libdevice -- segfaults inside NVVM. Forcing a
    matched nvcc wheel (cu11 or cu12) did not help there. PyTorch/CUDA is
    unaffected either way, since it never goes through numba.
"""
import numba
import numpy as np
from numba import njit

F32 = np.float32
_INLINE = dict(nopython=True, cache=True, inline="always")


@njit(**_INLINE)
def trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) -
            (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@njit(**_INLINE)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                         int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@njit(**_INLINE)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = np.zeros(2, F32)
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = np.zeros(2, F32)
        vs = np.zeros(64, F32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = np.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0.0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1
                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@njit(**_INLINE)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = np.zeros(2, F32)
    B = np.zeros(2, F32)
    C = np.zeros(2, F32)
    D = np.zeros(2, F32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]
    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]
    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]
    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]

    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@njit(**_INLINE)
def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]
    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]
    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]
    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1
    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@njit(**_INLINE)
def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = np.zeros(2, F32)
    for i in range(4):
        for j in range(4):
            if line_segment_intersection(pts1, pts2, i, j, temp_pts):
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1
    return num_of_inter


@njit(**_INLINE)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = np.cos(angle)
    a_sin = np.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = np.zeros(4, F32)
    corners_y = np.zeros(4, F32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = (-a_sin * corners_x[i] + a_cos * corners_y[i] +
                              center_y)


@njit(**_INLINE)
def inter(rbbox1, rbbox2):
    corners1 = np.zeros(8, F32)
    corners2 = np.zeros(8, F32)
    intersection_corners = np.zeros(64, F32)
    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)
    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    return area(intersection_corners, num_intersection)


@njit(**_INLINE)
def dev_rotate_iou_eval(rbox1, rbox2, criterion):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    return area_inter


@njit(parallel=True, cache=True)
def _rotate_iou_kernel(boxes, query_boxes, iou, criterion):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    for i in numba.prange(N):
        for j in range(K):
            # ARGUMENT ORDER IS DELIBERATE: the GPU kernel evaluates
            #   devRotateIoUEval(block_qboxes[i], block_boxes[tx], criterion)
            # i.e. the QUERY box is rbox1 and the indexed box is rbox2, even
            # though the output matrix is [N_boxes, K_queries]. criterion 0/1
            # ("area_inter/area1", "area_inter/area2") are asymmetric, so
            # passing (boxes[i], query_boxes[j]) here would silently swap them
            # -- verified against the real GPU kernel: it made crit0 match
            # crit1 and vice versa. criterion -1 and 2 are symmetric in this
            # argument, which is why they agree either way (and are the only
            # two mmdet3d actually uses: -1 for BEV, 2 for 3D).
            iou[i, j] = dev_rotate_iou_eval(query_boxes[j], boxes[i],
                                            criterion)


def rotate_iou_cpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Drop-in replacement for mmdet3d's rotate_iou_gpu_eval.

    `device_id` is accepted and ignored so the call sites need no change.
    """
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    query_boxes = np.ascontiguousarray(query_boxes, dtype=np.float32)
    N, K = boxes.shape[0], query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    _rotate_iou_kernel(boxes, query_boxes, iou, criterion)
    return iou


MODNAME = "mmdet3d.evaluation.functional.kitti_utils.rotate_iou"

# Minimal reproduction of the failure mode: an EAGER-signature cuda.jit, which
# forces NVVM to compile at decoration time. Importing the real rotate_iou
# would test the same thing but costs ~23 s (it compiles the full kernel);
# this costs ~2 s.
_PROBE = """
from numba import cuda
@cuda.jit('(float32[:],)')
def _p(a):
    i = cuda.grid(1)
    if i < a.size:
        a[i] = a[i]
print('PROBE_OK')
"""

_GPU_OK = None


def gpu_kernel_usable(timeout=180):
    """Can numba compile an eager-signature CUDA kernel on this host?

    Probed in a SUBPROCESS on purpose: a mismatched libnvvm/libdevice pairing
    segfaults the interpreter outright, and no try/except can catch a SIGSEGV
    in-process. Machine-dependent (a host with a complete /usr/local/cuda is
    fine; one that mixes a CUDA-12 libnvvm with a CUDA-11 libdevice is not),
    so the result is cached per process, not baked in.
    """
    global _GPU_OK
    if _GPU_OK is None:
        import subprocess
        import sys
        try:
            r = subprocess.run([sys.executable, "-c", _PROBE],
                               capture_output=True, timeout=timeout)
            _GPU_OK = (r.returncode == 0 and b"PROBE_OK" in r.stdout)
        except Exception:
            _GPU_OK = False
    return _GPU_OK


def patch(mode="auto", verbose=True):
    """Select the rotated-IoU backend for mmdet3d's KITTI evaluation.

    mode='auto' (default) uses mmdet3d's own GPU kernel where it compiles and
    falls back to the CPU port where it does not; 'gpu' and 'cpu' force one.
    Returns the backend actually selected.

    Both backends agree to float32 noise on every criterion (verify with
    `python kitti_iou_cpu.py`), so this choice affects speed, not results.

    For the CPU backend this installs a STUB MODULE in sys.modules rather than
    importing the real rotate_iou and reassigning its attribute. That
    distinction is the whole point: rotate_iou.py decorates its kernel with an
    eager signature, `@cuda.jit('(int64, int64, float32[:], ...)')`, so merely
    importing the module compiles the kernel -- and that compile is what
    segfaults. The real module must therefore never be imported at all.

    Both consumers (`bev_box_overlap`, `d3_box_overlap`) do a function-local
    `from .rotate_iou import rotate_iou_gpu_eval`, which resolves through
    sys.modules at call time and so picks up the stub. Call this BEFORE any
    KITTI evaluation runs. kitti_utils/__init__.py does not import rotate_iou,
    so importing mmdet3d first is safe.
    """
    import sys
    import types

    if mode == "auto":
        mode = "gpu" if gpu_kernel_usable() else "cpu"
    if mode == "gpu":
        if verbose:
            print("[kitti_iou_cpu] rotated IoU -> mmdet3d's own GPU kernel "
                  "(numba CUDA compiles on this host)")
        return "gpu"

    mod = sys.modules.get(MODNAME)
    if mod is None:
        mod = types.ModuleType(MODNAME)
        sys.modules[MODNAME] = mod
    mod.rotate_iou_gpu_eval = rotate_iou_cpu_eval
    mod.rotate_iou_cpu_eval = rotate_iou_cpu_eval

    parent = sys.modules.get(MODNAME.rsplit(".", 1)[0])
    if parent is not None:
        parent.rotate_iou = mod
    if verbose:
        print("[kitti_iou_cpu] rotated IoU -> CPU njit port "
              "(numba CUDA codegen unavailable on this host)")
    return "cpu"


# --- verification ------------------------------------------------------------
def _shapely_reference(boxes, qboxes, criterion=-1):
    """Independent reference using shapely, built from mmdet3d's own corner
    formula (clockwise rotation), to confirm the port is faithful."""
    from shapely.geometry import Polygon

    def corners(b):
        cx, cy, xd, yd, ang = [float(v) for v in b]
        ca, sa = np.cos(ang), np.sin(ang)
        lx = [-xd / 2, -xd / 2, xd / 2, xd / 2]
        ly = [-yd / 2, yd / 2, yd / 2, -yd / 2]
        return [(ca * x + sa * y + cx, -sa * x + ca * y + cy)
                for x, y in zip(lx, ly)]

    out = np.zeros((len(boxes), len(qboxes)), np.float32)
    for i, b in enumerate(boxes):
        pb = Polygon(corners(b))
        for j, q in enumerate(qboxes):
            pq = Polygon(corners(q))
            it = pb.intersection(pq).area
            if criterion == -1:
                den = pb.area + pq.area - it
            elif criterion == 0:
                den = pq.area      # rbox1 == the QUERY box (see _rotate_iou_kernel)
            elif criterion == 1:
                den = pb.area      # rbox2 == the indexed box
            else:
                den = 1.0
            out[i, j] = it / den if den > 0 else 0.0
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, k = 60, 40
    # KITTI-BEV-like boxes: centers in camera x/z, car-ish dims, any yaw
    boxes = np.stack([rng.uniform(-20, 20, n), rng.uniform(0, 70, n),
                      rng.uniform(1.4, 2.2, n), rng.uniform(3, 5, n),
                      rng.uniform(-np.pi, np.pi, n)], 1).astype(np.float32)
    q = np.stack([rng.uniform(-20, 20, k), rng.uniform(0, 70, k),
                  rng.uniform(1.4, 2.2, k), rng.uniform(3, 5, k),
                  rng.uniform(-np.pi, np.pi, k)], 1).astype(np.float32)
    # force many genuine overlaps: jitter copies of the first boxes
    q[:20] = boxes[:20] + rng.normal(0, .6, (20, 5)).astype(np.float32)

    print("--- CPU port vs independent shapely reference ---")
    for crit in (-1, 0, 1):
        mine = rotate_iou_cpu_eval(boxes, q, crit)
        ref = _shapely_reference(boxes, q, crit)
        d = np.abs(mine - ref).max()
        n_ov = int((ref > 1e-6).sum())
        print(f"criterion={crit:2d}  max|cpu-shapely|={d:.3e}  "
              f"overlapping pairs={n_ov}  -> "
              f"{'OK' if d < 2e-3 else 'MISMATCH'}")

    # Where the GPU kernel compiles, check the port against the thing it
    # actually replaces -- a stronger test than the shapely reference, since
    # it also pins the argument-order convention (criterion 0/1).
    print("--- CPU port vs mmdet3d's real GPU kernel ---")
    if not gpu_kernel_usable():
        print("GPU kernel does not compile on this host -- skipped "
              "(this is exactly the situation the CPU port exists for)")
    else:
        from mmdet3d.evaluation.functional.kitti_utils.rotate_iou import (
            rotate_iou_gpu_eval)
        for crit in (-1, 0, 1, 2):   # -1 = BEV, 2 = 3D: the two mmdet3d uses
            g = rotate_iou_gpu_eval(boxes, q, crit)
            c = rotate_iou_cpu_eval(boxes, q, crit)
            d = np.abs(g - c).max()
            print(f"criterion={crit:2d}  max|gpu-cpu|={d:.3e}  -> "
                  f"{'OK' if d < 2e-3 else 'MISMATCH'}")
