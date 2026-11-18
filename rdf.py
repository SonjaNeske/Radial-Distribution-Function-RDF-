"""Edge-corrected radial distribution function (RDF) for algae patch centers.

The input is a binary 2D grid of 0 and 1 in which each marked cell (1) represents an algae patch center. 
Grid spacings (dx and dy) are supplied in meters. 

The estimator averages the edge-corrected local neighbor densities 
and normalizes by the mean field density.
"""
import numpy as np
from scipy.spatial import cKDTree



def compute_rdf_from_binary(center_2d, dx, dy, dr=100.0, r_max=60000.0):
    """Compute an edge-corrected radial distribution function.

    Parameters
    ----------
    center_2d : 2D NumPy array, shape (ny, nx)
        Binary 2D matrix (1= Algae center; 0= no algae center).
    dx : 2D NumPy array, shape (ny, nx - 1)
        Horizontal distances between adjacent grid positions, in meters.
    dy : 2D NumPy array, shape (ny - 1, nx)
        Vertical distances between adjacent grid positions, in meters.
    dr : float, Radial bin width in meters (default: 100).
    r_max : float, Maximum separation in meters (default: 60000). 

    Returns
    -------
    r : ndarray, shape (n_bins,)
        Radial bin centers in meters.
    g_r : ndarray, shape (n_bins,)
        Dimensionless, edge-corrected radial distribution function.

    Notes
    -----
    The edge correction follows the nearest-boundary
    circular-segment/corner approximation. It assumes a rectangular domain
    with straight, perpendicular boundaries. The x-extent is evaluated at
    each center's row. For strongly varying
    row widths or radii that reach multiple opposite boundaries, this
    approximation may be inaccurate.

    The grid coordinates horizontal positions are accumulated separately along each row, while vertical positions
    use the first column of ``dy``. This assumes that vertical spacings are
    effectively constant across columns, as in the reference dataset.

    Example
    -------
    >>> r, g_r = compute_rdf_from_binary(center_2d, dx, dy, dr=100, r_max=60000)
    """

    # Convert input matrices to NumPy arrays if they are not already
    center_2d = np.asarray(center_2d)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    #---------------------------------------------------------------


    # Validate input dimensions and values
    if center_2d.ndim != 2:
        raise ValueError('center_2d must be a 2D array.')
    ny, nx = center_2d.shape
    if dx.shape != (ny, nx - 1) or dy.shape != (ny - 1, nx):
        raise ValueError('Expected dx.shape == (ny, nx-1) and dy.shape == (ny-1, nx).')
    if not np.all((center_2d == 0) | (center_2d == 1)):
        raise ValueError('center_2d must contain only 0 and 1.')
    if not (np.all(np.isfinite(dx)) and np.all(np.isfinite(dy)) and np.all(dx > 0) and np.all(dy > 0)):
        raise ValueError('dx and dy must be positive and finite.')
    if not (np.isfinite(dr) and np.isfinite(r_max) and dr > 0 and r_max > 0):
        raise ValueError('dr and r_max must be positive and finite.')
    n_bins = int(round(r_max / dr))
    if n_bins < 1 or not np.isclose(n_bins * dr, r_max, rtol=1e-10, atol=1e-8):
        raise ValueError('r_max must be an integer multiple of dr.')

    iy, ix = np.where(center_2d == 1)
    n = len(ix)
    if n < 2:
        raise ValueError('At least two patch centers are required.')
    #-----------------------------------------------------------------


    # Calculate spatial coordinates using cumulative distances in meters
    x_grid = np.concatenate((np.zeros((ny, 1)), np.cumsum(dx, axis=1)), axis=1)
    y_axis = np.concatenate(([0.0], np.cumsum(dy[:, 0])))
    x = x_grid[iy, ix]
    y = y_axis[iy]
    coords = np.column_stack((x, y))
    #------------------------------------------------------------------


    # Calculation of total area and total mean point density rho.
    area = np.sum(dx[:-1, :] * dy[:, :-1])
    rho = n / area
    #------------------------------------------------------------------

    # Calculation of the distances to the nearest horizontal and vertical boundaries.
    dist_x = np.minimum(x, x_grid[iy, -1] - x)[:, None]
    dist_y = np.minimum(y, y_axis[-1] - y)[:, None]
    #------------------------------------------------------------------


    # Calculation of radial bins and visible circular ring areas
    edges = np.linspace(0.0, r_max, n_bins + 1)
    inner = edges[:-1][None, :]
    outer = edges[1:][None, :]
    r = 0.5 * (edges[:-1] + edges[1:])
    full_ring = np.pi * (outer**2 - inner**2)

    cut_x = segment_area_circle(outer, dist_x) - segment_area_circle(inner, dist_x)
    cut_y = segment_area_circle(outer, dist_y) - segment_area_circle(inner, dist_y)
    corner = corner_area_circle(outer, dist_x, dist_y) - corner_area_circle(inner, dist_x, dist_y)
    visible_area = np.maximum(full_ring - cut_x - cut_y + corner, 0.0)
    #------------------------------------------------------------------


    # Counting of neighboring patch centers in each radial bin
    counts = np.zeros((n, n_bins), dtype=np.int32)
    tree = cKDTree(coords)
    for p in range(n):
        neighbors = tree.query_ball_point(coords[p], r_max)
        distances = np.linalg.norm(coords[neighbors] - coords[p], axis=1)
        distances = distances[distances > 0]
        counts[p], _ = np.histogram(distances, bins=edges)
    #-----------------------------------------------------------------


    # Calculation of the RDF by averaging local densities over all reference points
    if np.any(visible_area <= 0):
        raise ValueError('Zero visible ring area encountered; reduce r_max or review domain geometry.')
    
    g_r = np.mean(counts / visible_area, axis=0) / rho
    return r, g_r
    #----------------------------------------------------------------



def segment_area_circle(r, d):
    """
    Calculates the area of a circular segment extending beyond a boundary.

    Parameters
    ----------
    r : float or NumPy array
        Circle radius.
    d : float or NumPy array
        Distance from the circle center to the boundary.

    Returns
    -------
    NumPy array
        Area of the circular segment outside the boundary.
    """
    val = (
        r**2 * np.arccos(np.clip(d / r, -1, 1))
        - d * np.sqrt(np.maximum(r**2 - d**2, 0))
    )

    return np.where(d < r, val, 0.0)



def corner_area_circle(r, dx, dy):
    """
    Calculate the overlap area of two circular segments extending
    beyond perpendicular boundaries.

    Returns zero if the circle does not reach the corner.

    Parameters
    ----------
    r : float or NumPy array
        Circle radius.
    dx : float or NumPy array
        Distance to the nearest vertical boundary.
    dy : float or NumPy array
        Distance to the nearest horizontal boundary.

    Returns
    -------
    NumPy array
        Overlap area outside both boundaries..
    """
    out = np.zeros(np.broadcast_shapes(np.shape(r), np.shape(dx), np.shape(dy)))

    rr = np.broadcast_to(r, out.shape)
    dxx = np.broadcast_to(dx, out.shape)
    dyy = np.broadcast_to(dy, out.shape)

    mask = dxx**2 + dyy**2 < rr**2

    x0 = dxx
    x1 = np.zeros_like(out)
    x1[mask] = np.sqrt(rr[mask]**2 - dyy[mask]**2)

    def F(x, r):
        return 0.5 * (
            x * np.sqrt(np.maximum(r**2 - x**2, 0))
            + r**2 * np.arcsin(np.clip(x / r, -1, 1))
        )

    out[mask] = (
        F(x1[mask], rr[mask])
        - F(x0[mask], rr[mask])
        - dyy[mask] * (x1[mask] - x0[mask])
    )

    return out

      
