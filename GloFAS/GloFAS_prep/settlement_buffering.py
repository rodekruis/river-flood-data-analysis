import GloFAS.GloFAS_prep.configuration as cfg 
import rasterio
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape, box
from shapely.ops import unary_union
from rasterio.features import rasterize
import xarray as xr

def settlements_buffered(settlements_tif, buffer_distance):
    """
    Creates a binary xarray.DataArray with a buffer around settlements using an optimized workflow.
    
    Parameters:
        settlements_tif (str): Path to the raster file containing settlements data.
        buffer_distance (float): Buffer distance in metres.
        
    Returns:
        xarray.DataArray: A binary raster with the buffer applied.
    """
    # read the raster
    with rasterio.open(settlements_tif) as src:
        settlements = src.read(1)  # Read the first band
        transform = src.transform
        resolution = abs(transform[0])  # Assuming square pixels
        crs = src.crs
        profile = src.profile

    # create a binary mask (values > 0)
    binary_mask = (settlements > 0).astype(np.uint8)

    # extract settlement shapes
    settlement_shapes = [
        shape(geom) for geom, value in shapes(binary_mask, transform=transform) if value > 0
    ]

    # buffer the shapes
    buffered_shapes = [geom.buffer(buffer_distance) for geom in settlement_shapes]

    # Combine all buffered geometries into one
    unified_buffer = unary_union(buffered_shapes)

    # rasterize the buffered geometry
    buffered_raster = rasterize(
        [(unified_buffer, 1)],  # Assign value 1 to the buffered geometry
        out_shape=settlements.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    # convert to xarray.DataArray
    coords = {
        "y": np.arange(profile['height']) * transform[4] + transform[5],
        "x": np.arange(profile['width']) * transform[0] + transform[2],
    }
    buffered_da = xr.DataArray(buffered_raster, dims=["y", "x"], coords=coords)

    return buffered_da
# also use this one for the threshold Q_da: 
def multiply(Q_da, settlement_da):
    """
    Multiply Q_da with settlement_da so that Q values only show up where there is a settlement.
    
    Parameters:
        Q_da (xarray.DataArray): Coarser resolution DataArray containing Q values.
        settlement_da (xarray.DataArray): Finer resolution DataArray with buffered settlements (binary).
        
    Returns:
        xarray.DataArray: Q values masked by settlements.
    """
    # Resample settlement_da to match Q_da's resolution
    settlement_resampled = settlement_da.coarsen(
        x=int(settlement_da.sizes['x'] / Q_da.sizes['x']),
        y=int(settlement_da.sizes['y'] / Q_da.sizes['y'])
    ).mean()
    
    # Convert resampled settlements to binary (threshold 0.5)
    settlement_resampled = (settlement_resampled > 0.5).astype(Q_da.dtype)
    
    # Multiply Q_da with settlement mask
    Q_at_settlements_da = Q_da * settlement_resampled
    
    return Q_at_settlements_da

if __name__=='__main__':
    settlements_tif = cfg.settlements_tif
    buffer_distance = 5000              # m
    buffered_da = settlements_buffered (settlements_tif, buffer_distance)
    buffered_da.plot()


