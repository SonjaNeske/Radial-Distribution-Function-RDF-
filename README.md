# Radial Distribution Function (RDF) with Edge Correction

This repository provides a Python function to calculate the radial distribution function (RDF) of points in a two-dimensional binary matrix (1 = point, 0 = no point).

The function was originally developed to analyze the spatial distribution of macroalgal patch centers represented in a 2D binary matrix (`center_2d`).

The RDF quantifies the spatial distribution of points as a function of their separation distance. A geometric edge correction accounts for circular ring areas extending beyond the observation boundaries.

## Requirements

- Python 3
- NumPy
- SciPy

Install the required packages using:

```bash
pip install numpy scipy
```

## Input Data

The function requires three input arrays:

- `center_2d`: Binary 2D array, where `1` indicates the presence of a point and `0` indicates its absence.
- `dx`: Horizontal distances between adjacent grid positions, in meters.
- `dy`: Vertical distances between adjacent grid positions, in meters.

The arrays must have the following shapes:

- `center_2d`: `(ny, nx)`
- `dx`: `(ny, nx - 1)`
- `dy`: `(ny - 1, nx)`

## Usage

```python
from rdf import compute_rdf_from_binary

r, g_r = compute_rdf_from_binary(
    center_2d,
    dx,
    dy,
    dr=100,
    r_max=60000
)
```

Here, `dr` is the radial bin width in meters, and `r_max` is the maximum separation distance in meters.

## Output

The function returns:

- `r`: Radial bin centers in meters.
- `g_r`: Dimensionless, edge-corrected radial distribution function.

## Method

The RDF is calculated by counting neighboring points within radial distance bins. The local neighbor density is corrected for the visible circular ring area and averaged over all reference points. The result is normalized by the mean point density of the observation domain.

The geometric edge correction uses circular segments and corner overlaps, assuming an approximately rectangular observation domain. The implementation uses the nearest vertical and horizontal boundaries and may be less accurate when radial distances reach multiple opposite boundaries.
