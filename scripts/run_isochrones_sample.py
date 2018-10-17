# Standard library
import os
from os import path
import sys

# Third-party
from astropy.table import Table, hstack
import astropy.units as u
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from isochrones import StarModel
from isochrones.observation import Source, Observation, ObservationTree
from isochrones.mist import MIST_Isochrone
from isochrones.priors import FlatPrior, PowerLawPrior

def load_data(filename):
    decam = Table.read(filename)
    gi = decam['GMAG'] - decam['IMAG']
    gmag = decam['GMAG']

    cmd_mask = (gi < 0.3) & (gi > -1) & (gmag < 22) & (gmag > 15)
    return decam[cmd_mask]


def main(index, overwrite=False):
    # First make sure paths exist:
    # os.makedirs('../cache', exist_ok=True)
    # os.makedirs('../plots/isochrones', exist_ok=True)

    # Load the DECam photometry
    decam = load_data('../data/decam_apw.fits')

    iso = MIST_Isochrone(['DECam_u',
                          'DECam_g',
                          'DECam_i'])

    row = decam[index]
    name = 'lmcla-{0}-'.format(row['index'])
    model_file = '../cache/starmodels-{0}.hdf5'.format(str(row['index']))

    if path.exists(model_file) and not overwrite:
        print('skipping {0} - already exists'.format(name))
        sys.exit(0)

    # This is our "anchor star": it was identified as being near the turnoff,
    # bright, and positionally consistent with being in the LA cluster:
    j1, = np.where(decam['index'] == 24365)[0]
    j2 = row['index']

    # To fit pairs as resolved binaries, we have to construct the observation
    # tree manually:
    tree = ObservationTree()
    for b in ['DECam_g', 'DECam_i']:
        o = Observation('DECam', b, 1.)
        s0 = Source(decam[b[-1].capitalize() + 'MAG'][j1],
                    np.sqrt(0.01**2 + decam[b[-1].capitalize() + 'ERR'][j1]**2))
        s1 = Source(decam[b[-1].capitalize() + 'MAG'][j2],
                    np.sqrt(0.01**2 + decam[b[-1].capitalize() + 'ERR'][j2]**2),
                    separation=100.)
        o.add_source(s0)
        o.add_source(s1)
        tree.add_observation(o)

    model = StarModel(ic=iso, obs=tree)

    print('setting priors')
    model.set_bounds(distance=(1000., 100000.)) # 1 to 100 kpc
    model.set_bounds(eep=(202, 355)) # ZAMS to TAMS

    model.set_bounds(feh=(-2, 0.5))
    model._priors['feh'] = FlatPrior((-2, 0.5))

    model.set_bounds(AV=(1e-3, 1))
    model._priors['AV'] = PowerLawPrior(-1.1, (1e-3, 1))

    model.set_bounds(mass=(0.2, 20.))

    print('sampling star {0}'.format(row['index']))
    model.fit_multinest(basename=name, overwrite=overwrite)
    # model.fit_mcmc(nwalkers=nwalkers,
    #                p0=np.array([350., 8., -0.5, 30000., 0.1]),
    #                nburn=1024, niter=2048)
    model.save_hdf(model_file)

    fig = model.corner_physical()
    fig.savefig('../plots/isochrones/{0}-physical.png'.format(row['index']),
                dpi=200)
    plt.close(fig)

    fig = model.corner_observed()
    fig.savefig('../plots/isochrones/{0}-observed.png'.format(row['index']),
                dpi=200)
    plt.close(fig)

    # model._samples = model.samples[::1024]
    # model.save_hdf(sm_model_file)

    sys.exit(0)


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    # vq_group = parser.add_mutually_exclusive_group()
    # vq_group.add_argument('-v', '--verbose', action='count', default=0,
    #                       dest='verbosity')
    # vq_group.add_argument('-q', '--quiet', action='count', default=0,
    #                       dest='quietness')

    # parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
    #                     help="Random number seed")

    parser.add_argument("-i", "--index", dest="index", required=True, type=int)

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action='store_true')

    args = parser.parse_args()

    main(index=args.index, overwrite=args.overwrite)
