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
from isochrones.mist import MIST_Isochrone

from schwimmbad import choose_pool

def load_data(filename):
    decam = Table.read(filename)
    gi = decam['GMAG'] - decam['IMAG']
    gmag = decam['GMAG']
    decam['index'] = np.arange(len(decam), dtype=int)

    cmd_mask = (gi < 0.3) & (gi > -1) & (gmag < 22) & (gmag > 15)
    return decam[cmd_mask]


def worker(task):
    i1, i2, overwrite = task

    decam = load_data('../data/decam_apw.fits')
    iso = MIST_Isochrone(['DECam_u',
                          'DECam_g',
                          'DECam_i'])
    nwalkers = 300 # MAGIC NUMBER

    for i in np.arange(i1, i2+1, dtype=int):
        row = decam[i]
        name = 'lmcla-{0}-'.format(row['index'])
        model_file = '../cache/starmodels-{0}.hdf5'.format(str(row['index']))
        sm_model_file = '../cache/starmodels-sm-{0}.hdf5'.format(str(row['index']))

        if path.exists(model_file) and not overwrite:
            print('skipping {0} - already exists'.format(name))
            continue

        model = StarModel(ic=iso,
                          DECam_g=(row['GMAG'], 2 * row['GERR']), # MAGIC NUMBER
                          DECam_i=(row['IMAG'], 2 * row['IERR']), # MAGIC NUMBER
                          use_emcee=True)

        print('sampling star {0}'.format(row['index']))
        model.set_bounds(distance=(1000., 50000.))
        # model.fit_multinest(basename=name, overwrite=overwrite)
        model.fit_mcmc(nwalkers=nwalkers,
                       p0=np.array([350., 8., -0.5, 30000., 0.1]),
                       nburn=1024, niter=2048)
        model.save_hdf(model_file)

        fig = model.corner_physical()
        fig.savefig('../plots/isochrones/{0}-physical.png'.format(row['index']),
                    dpi=200)
        plt.close(fig)

        fig = model.corner_observed()
        fig.savefig('../plots/isochrones/{0}-observed.png'.format(row['index']),
                    dpi=200)
        plt.close(fig)

        model._samples = model.samples[::1024]
        model.save_hdf(sm_model_file)


def main(pool, n_cores=24, overwrite=False):
    from helpers import chunk_tasks

    decam = load_data('../data/decam_apw.fits')
    n_rows = len(decam)
    print('{0} stars to process'.format(n_rows))

    tasks = [(i1, i2, overwrite) for (i1, i2), _ in chunk_tasks(n_rows, n_cores-1)]

    # First make sure paths exist:
    os.makedirs('../cache', exist_ok=True)
    os.makedirs('../plots/isochrones', exist_ok=True)

    for _ in pool.map(worker, tasks):
        pass

    pool.close()
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

    parser.add_argument("--ncores", dest="n_cores", default=24, type=int,
                        help="Number of MPI threads")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)

    main(pool=pool, overwrite=args.overwrite, n_cores=args.n_cores)
