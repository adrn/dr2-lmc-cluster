"""
Generate posterior samples in age, metallicity, distance for one of the youngest
stars that we think is in the cluster. We'll use the posterior samples from this
to set the prior for all other stars.
"""

# Standard library
from os import path
import sys

# Third-party
import numpy as np
import matplotlib.pyplot as plt

from isochrones import StarModel
from isochrones.mist import MIST_Isochrone
from isochrones.priors import FlatPrior, PowerLawPrior

from .helpers import load_data

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
    j, = np.where(decam['index'] == 24365)[0]
    row = decam[j]

    model = StarModel(ic=iso,
                      DECam_g=(row['GMAG'], 2 * row['GERR']), # MAGIC NUMBER
                      DECam_i=(row['IMAG'], 2 * row['IERR']), # MAGIC NUMBER
                      use_emcee=True)

    print('setting priors')
    model.set_bounds(distance=(1000., 100000.)) # 1 to 100 kpc
    model.set_bounds(eep=(202, 355)) # ZAMS to TAMS

    model.set_bounds(feh=(-2, 0.5))
    model._priors['feh'] = FlatPrior((-2, 0.5))

    model.set_bounds(AV=(1e-3, 1))
    model._priors['AV'] = PowerLawPrior(-1.1, (1e-3, 1))

    model.set_bounds(mass=(0.02, 25.))

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

    sys.exit(0)


if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action='store_true')

    args = parser.parse_args()

    main(index=args.index, overwrite=args.overwrite)
