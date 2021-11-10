import xarray as xr
import numpy as np
import dask
import sys
import graphviz

# note that in my comments I do NOT use 0-indexing.

# Scheduling

# Dask core objects are collections (dask array, dask data frame dask bag, and so on). Whenever you operate on these,
# dask creates a dask graph that describes the operation. A node is a function (an operation), and edges are inputs or
# outputs.
#
# This graph is a representation of what's to be done, and it can be done in multiple ways, that's what we call
# a scheduling type. Scheduling types can be either : single machine scheduler, or distributed scheduler. Names are self
# explanatory. Depending on how you use this stuff, you will be doing varying types of multi threading,
# multi processing, or no parallelism at all.

# For now, let's turn off any parallelism on our local machine

dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler


# Dask collections

# creating a fake array
# time (say, year)
time = range(1, 10001)
# space (say, IR)
space = range(1, 10001)
# temperature value (Gaussian distribution with mean=15) ~ 1GB
temperature_array = np.random.randn(len(time), len(space)) + 15

# Chunking your dask collection -- splitting it into pieces ('chunking')

# Method 1
dask_array = dask.array.from_array(temperature_array, chunks=(1000)) # cutting into 10 arrays each of shape [1000, 1000]
dask_array.numblocks == (10,10)
dask_array.npartitions == (100)
dask_array.chunksize == (1000, 1000)
dask_array.blocks[0,0].shape == (1000, 1000)
dask_array.blocks[0,1][0,0].compute() == temperature_array[0,1000]  # LHS expression gives you the 1st row, 2nd column block, first value.
dask_array.blocks[4,7][0,0].compute() == temperature_array[4000,7000]  # LHS expression gives you the 5th row, 8th column block, first value.

# Method 2
dask_array = dask.array.from_array(temperature_array, chunks=((7000, 3000), (2000, 2000, 6000))) # Let's try blocks of different size
dask_array.npartitions == (6)
dask_array.numblocks == (2, 3)
dask_array.chunksize ==(7000, 6000) # this gives you the maximum size across blocks for each dimensions of the underlying array
dask_array.blocks[0,0].shape == (7000, 2000)
dask_array.blocks[1,2].shape == (3000, 6000)
dask_array.blocks[1,2][0,0].compute() == temperature_array[7000,4000]

# Method 3
dask_array = dask.array.from_array(temperature_array, chunks="auto")
dask_array.numblocks == (4,4)
dask_array.chunksize == (2500, 2500)
dask_array.blocks[1,1].shape == (2500, 2500)
dask_array.blocks[1,1][0,0].compute() == temperature_array[2500,2500]


# Indexing your dask collection

dask_array[0,0].compute() == temperature_array[0,0]
dask_array[5424,3201].compute() == temperature_array[5424,3201]

# Computing over your dask collection

a = dask_array.sum()
b = dask_array.mean()
c = a + b # nothing is done yet !
c.compute() # now it's done

# Visualizing the graph of operations involving your dask collection

a.dask.visualize().render(view=True)
b.dask.visualize().render(view=True)
c.dask.visualize().render(view=True)


# Delaying on arbitrary python functions and objects

@dask.delayed
def multiply(a, b):
    return a*b

c = multiply(1,2)
d = multiply(c, 3)
d.dask.visualize().render(view=True)
d.compute()