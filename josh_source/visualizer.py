"""
skeleton3d.py
=============

Visualize a 3D skeleton with matplotlib.

Typical use
-----------
    from skeleton3d import plot_skeleton
    fig, ax = plot_skeleton(coords)            # coords: (N, 3) or (N, 4) array
    # or supply your own weights / connectivity:
    fig, ax = plot_skeleton(coords, node_weights=my_weights, edges=my_edges)

How nodes map to rows
---------------------
`node_weights` is an *ordered* dict describing the full skeleton. Any node with
weight 0 is dropped from the figure. The remaining (non-zero) nodes, taken in
dict order, are matched one-to-one against the rows of `coords`. So `coords`
must have exactly as many rows as there are non-zero weights.

Only the first three columns of `coords` are used as (x, y, z); a 4th column
(confidence / homogeneous w / etc.) is ignored.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)


# --- Default skeleton definition (rodent-style pose layout) -----------------
DEFAULT_NODE_WEIGHTS = {
    'Nose':           0.7,
    'Ear_R':          0.7,
    'Ear_L':          0.7,
    'TTI':            1,
    'TailTip':        0,
    'Head':           1,
    'Trunk':          0.8,
    'Tail_0':         0,
    'Tail_1':         0,
    'Tail_2':         0,
    'Shoulder_left':  0.7,
    'Shoulder_right': 0.7,
    'Haunch_left':    0.7,
    'Haunch_right':   0.7,
    'Neck':           0.7,
}

# Bones are defined by node *name* (independent of array order). A bone is only
# drawn when both of its endpoints survived the weight filter, so edges that
# touch a removed node (e.g. the tail chain) simply disappear automatically.
DEFAULT_EDGES = [
    ('Nose', 'Head'),
    ('Head', 'Ear_R'),
    ('Head', 'Ear_L'),
    ('Head', 'Neck'),
    ('Neck', 'Shoulder_left'),
    ('Neck', 'Shoulder_right'),
    ('Neck', 'Trunk'),
    ('Trunk', 'Haunch_left'),
    ('Trunk', 'Haunch_right'),
    ('Trunk', 'TTI'),
    ('TTI', 'Tail_0'),
    ('Tail_0', 'Tail_1'),
    ('Tail_1', 'Tail_2'),
    ('Tail_2', 'TailTip'),
]


def _set_axes_equal(ax):
    """Give the 3 axes the same scale so the skeleton isn't distorted."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = 0.5 * (limits[:, 1] - limits[:, 0]).max()
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def kept_nodes(node_weights):
    """Return the ordered list of node names with non-zero weight."""
    return [name for name, w in node_weights.items() if w != 0]


def map_coords(coords, node_weights):
    """
    Build a {node_name: (x, y, z)} dict by matching non-zero-weight nodes
    (in dict order) to the rows of `coords`.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(
            f"coords must be 2D with >= 3 columns, got shape {coords.shape}"
        )

    names = kept_nodes(node_weights)
    if len(names) != coords.shape[0]:
        raise ValueError(
            f"Number of non-zero weights ({len(names)}) does not match the "
            f"number of coordinate rows ({coords.shape[0]}). "
            f"Non-zero nodes (in order): {names}"
        )

    return {name: coords[i, :3] for i, name in enumerate(names)}


def plot_skeleton(
    coords,
    node_weights=DEFAULT_NODE_WEIGHTS,
    edges=DEFAULT_EDGES,
    ax=None,
    annotate=True,
    scale_markers_by_weight=True,
    bone_color="steelblue",
    joint_color="crimson",
    view=(15, -70),
    show=True,
    save_path=None,
):
    """
    Plot a 3D skeleton.

    Parameters
    ----------
    coords : array-like, shape (N, 3) or (N, 4)
        Joint positions. Rows correspond to the non-zero-weight nodes of
        `node_weights`, in order. Only the first 3 columns are used.
    node_weights : dict
        Ordered {name: weight}. Weight 0 removes the node (and its bones).
    edges : list of (name, name)
        Bone connectivity by node name. Bones touching a removed node are
        skipped automatically.
    ax : mpl 3d Axes, optional
        Draw onto an existing axes (e.g. for animation). A new figure is
        created if None.
    annotate : bool
        Label each joint with its node name.
    scale_markers_by_weight : bool
        Scale joint marker size by its weight (purely cosmetic).
    bone_color, joint_color : str
        Colors for bones and joints.
    view : (elev, azim)
        Camera angle.
    show : bool
        Call plt.show() at the end.
    save_path : str, optional
        If given, save the figure to this path.

    Returns
    -------
    (fig, ax)
    """
    pos = map_coords(coords, node_weights)

    if ax is None:
        fig = plt.figure(figsize=(7, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # --- bones ---
    for a, b in edges:
        if a in pos and b in pos:
            pa, pb = pos[a], pos[b]
            ax.plot(
                [pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                color=bone_color, linewidth=2.5, zorder=1,
            )

    # --- joints ---
    names = list(pos.keys())
    pts = np.array([pos[n] for n in names])
    if scale_markers_by_weight:
        sizes = np.array([40 + 60 * node_weights[n] for n in names])
    else:
        sizes = 50
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        color=joint_color, s=sizes, depthshade=True, zorder=2,
    )

    # --- labels ---
    if annotate:
        for n in names:
            x, y, z = pos[n]
            ax.text(x, y, z, f"  {n}", fontsize=7, color="black")

    _set_axes_equal(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=view[0], azim=view[1])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    # Sample data: 11 rows == 11 non-zero weights in DEFAULT_NODE_WEIGHTS
    sample = np.array([
        [-3.55255498e-02,  4.07469818e-02, -9.98537396e-01, -8.29002483e-04],
        [ 7.84700375e-05,  4.33526145e-02, -9.99059462e-01, -8.57736775e-04],
        [-1.59572595e-02,  7.15304786e-02, -9.97310398e-01, -8.52361758e-04],
        [ 4.95354819e-02,  8.79396002e-02, -9.94893075e-01, -7.95346558e-04],
        [-1.29230512e-02,  5.46527167e-02, -9.98421428e-01, -8.52548439e-04],
        [ 2.47244077e-02,  8.95144384e-02, -9.95678239e-01, -8.44837088e-04],
        [-2.13428106e-02,  6.93247644e-02, -9.97365478e-01, -8.14909302e-04],
        [-1.38218838e-03,  3.80717999e-02, -9.99273705e-01, -8.30141617e-04],
        [ 1.07487165e-02,  1.01510082e-01, -9.94776125e-01, -7.93211868e-04],
        [ 3.60720073e-02,  5.47045779e-02, -9.97850466e-01, -8.16594052e-04],
        [-2.23518891e-03,  6.15108997e-02, -9.98103545e-01, -8.52365019e-04],
    ])
    plot_skeleton(sample)