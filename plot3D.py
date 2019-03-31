from numpy import array, r_, log10, absolute, floor
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

def plot_trimesh(verts, faces, val=None, cmap=plt.cm.get_cmap(), 
        vmin=None, vmax=None, **kw):
    """Doesn't work."""
    raise NotImplementedError
    
    """Plot a mesh of triangles.
    
    Input   verts   N x 3 array of vertex locations
    
            faces   M x 3 array of vertex indices for each face
            
            val     M list of values for each vertex, used for coloring
            
            cmap    colormap, defaulting to current colormap
            
            vmin    lower limit for coloring, defaulting to min(val)
            
            vmax    upper limit for coloring, defaulting to max(val)
            
            Other keyword pairs are passed to Poly3DCollection
    """
    
    v = [[verts[ind,:] for ind in face] for face in faces]
    if val is not None:
        val = array(val)  # To be safe - set_array() will expect a numpy array
        
    poly = Poly3DCollection(v, cmap=cmap, norm=Normalize(clip=True), **kw)
    # Have to set clip=True in Normalize for when vmax < max(val)
    poly.set_array(val)  # sets vmin, vmax to min, max of val
    poly.set_clim(vmin, vmax) # sets vmin, vmax if not None
    poly.set_facecolors(poly.norm(val)) 
    # if val > 1, can't set with facecolor = val in definition.
    # Isn't that bizarre?
    
    ax = plt.gca()
    if type(ax) is not Axes3D:
        ax = Axes3D(plt.figure()) # Make a new figure and a new axes
    ax.add_collection3d(poly)
    ax.auto_scale_xyz(verts[:,0], verts[:,1], verts[:,2])
    if val is not None:
        # This magic dohickey is used by colorbar() and clim(), for example
        plt.gci._current = poly
        
    return poly
    

def plot_trimesh2D(verts, faces, val=None, cmap=plt.cm.get_cmap(), 
        vmin=None, vmax=None, mirror=False, **kw):
    """Plot a mesh of triangles, from directly above, without all this
    3D stuff.
    
    Input   verts   N x 3 array of vertex locations
    
            faces   M x 3 array of vertex indices for each face
            
            val     M list of values for each vertex, used for coloring
            
            cmap    colormap, defaulting to current colormap
            
            vmin    lower limit for coloring, defaulting to min(val)
            
            vmax    upper limit for coloring, defaulting to max(val)
            
            mirror  flag for mirror around x=0
            
            Other keyword pairs are passed to PolyCollection
    """
    
    v = array([[verts[ind,:2] for ind in face] for face in faces])
    if mirror:
        v = r_[v, v * [-1,1]]
    if val is not None:
        val = array(val)
    
    poly = PolyCollection(v, cmap=cmap, norm=Normalize(clip=True), **kw)
    poly.set_array(val)
    poly.set_clim(vmin, vmax)
    poly.set_facecolors(poly.norm(val)) 
    
    ax = plt.gca()
    ax.add_collection(poly)
    ax.axis('image')
    if val is not None:
        # This magic dohickey is used by colorbar() and clim(), for example
        plt.gci._current = poly
    plt.draw() # Seems like should be draw_if_interactive(), but that doesn't
               # work, for some unexplained reason.
    
    return poly

def plot_trimesh2D_cb(verts, faces, val=None, cmap=plt.cm.get_cmap(), 
        vmin=None, vmax=None, label="", **kw):
    """Plot a mesh of triangles, from directly above, without all this
    3D stuff.  Includes a handy colorbar.
    
    Input   verts   N x 3 array of vertex locations
    
            faces   M x 3 array of vertex indices for each face
            
            val     M list of values for each vertex, used for coloring
            
            cmap    colormap, defaulting to current colormap
            
            vmin    lower limit for coloring, defaulting to min(val)
            
            vmax    upper limit for coloring, defaulting to max(val)
            
            label   the label for the colorbar
            
            Other keyword pairs are passed to PolyCollection
    """
    
    if val is not None:
        val = array(val)
        if vmax: # Catch None and 0 as being inappropriate
            nm = floor(log10(abs(vmax)))
            if vmin:
                n = max([nm, floor(log10(abs(vmin)))])
            else:
                n = nm
        elif vmin:
            n = floor(log10(abs(vmin)))
        else:
            n = floor(log10(max(absolute(val))))
        if n < -2 or n > 2:
            val *= 10**-n
            if vmin is not None:
                vmin *= 10**-n
            if vmax is not None:
                vmax *= 10**-n
            label += " x 10^%i"%(-n)
    
    p = plot_trimesh2D(verts, faces, val, cmap, vmin, vmax, **kw)
    if val is not None:
        cb = plt.colorbar(p, orientation='horizontal')
        for l in cb.ax.get_xticklabels():
            l.update({'rotation': -60, 'horizontalalignment': 'left'})
        cb.set_label(label)
        plt.draw()
        
    return p
