import numpy as np
import functools as ftools

from logzero import logger

try:
    from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn
except:
    from numpy.fft import fftshift, fftn, ifftn

def fftn_n( arr ):
    return fftn( arr, norm='ortho' )

def ifftn_n( arr ):
    return ifftn( arr, norm='ortho' )


chirp = np.mgrid[ 0:1, 0:1, 0:1 ]
chirp_arg = 1.j * np.pi * ftools.reduce( lambda x, y: x+y, chirp )

pref0 = 'chirp = tuple( fftshift( this )**2 / this.shape[n] for n, this in enumerate( np.mgrid[ '
suff0 = ' ] ) )'

DoNothing = lambda x: x
opdict = { 0:DoNothing, 1:fftn_n, 2:np.flip, 3:ifftn_n }

def frft( arr, alpha ):
    if arr.shape != chirp[0].shape:
        RecalculateChirp( arr.shape, arr )
    ops = CanonicalOps( alpha )
    return frft_base( ops[0]( arr ), ops[1] )

def frft_base( arr, alpha ):
    #if arr.shape != chirp_arg.shape:
    #    arr = np.delete(chirp_arg, (0), axis=0)
    #    print("a")
    #print(arr.shape)
    #print(chirp_arg.shape)
    phi = alpha * np.pi/2.
    cotphi = 1. / np.tan( phi )
    cscphi = np.sqrt( 1. + cotphi**2 )
    scale = np.sqrt( 1. - 1.j*cotphi ) / np.sqrt( np.prod( arr.shape ) )
    modulator = ChirpFunction( cotphi - cscphi )
    filtor = ChirpFunction( cscphi )
    #print(modulator.shape)
    #print(arr.shape)
    inc = 0
    if inc == 2:
        modulator = np.delete(chirp_arg, (0), axis=0)
        inc = 0
    else:
        inc += 1
    #print(inc)
    #print(modulator.shape)
    arr_frft = scale * modulator * ifftn_n( fftn_n( filtor ) * fftn_n( modulator * arr ) )
    return arr_frft

def ChirpFunction( x ):
    return np.exp( x * chirp_arg )

def RecalculateChirp( newshape, arr ):
    logger.warning( 'Recalculating chirp. ' )
    global chirp_arg
    if len( newshape ) == 1:    # extra-annoying string manipulations needed with 1D data
        pref = pref0.replace( 'np.', '( np.' )
        suff = suff0.replace( ']', '], )' )
    else:
        pref = pref0
        suff = suff0
    regrid = ','.join( tuple( '-%d:%d'%(n//2,n//2) for n in newshape ) ).join( [ pref, suff ] )
    #print( regrid )
    exec( regrid, globals() )
    chirp_arg = 1.j * np.pi * ftools.reduce( lambda x, y: x+y, chirp )
    print(chirp_arg.shape)

    new_col = chirp_arg.sum(1)[..., None]
    all_data = np.hstack((chirp_arg, new_col))
    if chirp_arg.shape[0] < arr.shape[0]:
        blank_row = np.zeros((1, all_data.shape[1]))
        new_array = np.vstack([all_data, blank_row])
        chirp_arg = new_array
    else:
        chirp_arg = all_data
    print(chirp_arg.shape)
    print(arr.shape)

    return

def CanonicalOps( alpha ):
    alpha_0 = alpha % 4.
    if alpha_0 < 0.5:
        return[ ifftn_n, 1.+alpha_0 ]
    flag = 0
    while alpha_0 > 1.5:
        alpha_0 -= 1.
        flag += 1
    return [ opdict[flag], alpha_0 ]