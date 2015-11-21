import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from scipy.stats import spearmanr
from lsst.sims.maf.utils.astrometryUtils import m52snr, astrom_precision

__all__ =['ParallaxCoverageMetric','ParallaxHADegenMetric']

class ParallaxCoverageMetric(BaseMetric):
    """
    Check how well the parallax factor is distributed.
    """
    def __init__(self, metricName='ParallaxCoverageMetric', m5Col='fiveSigmaDepth',
                 mjdCol='expMJD', filterCol='filter', seeingCol='finSeeing',
                 rmag=20., SedTemplate='flat', badval=-666,
                 atm_err=0.01,**kwargs):

        cols = ['ra_pi_amp', 'dec_pi_amp', m5Col, mjdCol, filterCol, seeingCol]
        units = 'ratio'
        super(ParallaxCoverageMetric, self).__init__(cols,
                                                     metricName=metricName, units=units,
                                                      **kwargs)
        self.m5Col = m5Col
        self.seeingCol = seeingCol
        self.filterCol = filterCol
        self.mjdCol = mjdCol

        filters=['u','g','r','i','z','y']
        self.mags={}
        if SedTemplate == 'flat':
            for f in filters:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(SedTemplate, rmag=rmag)
        self.atm_err = atm_err


    def _aveFactor(self, ra_pi_amp, dec_pi_amp, weights):
        """
        Find the average parallax factor:
        1 is good, 0 is bad
        """
        radius = np.sqrt(ra_pi_amp**2 + dec_pi_amp**2 )
        # weights are for ra or dec, so the quad sum means...
        # XXX
        return np.average(radius, weights=weights)

    def _avePosition(self, ra_pi_amp, dec_pi_amp, weights):
        """
        Find the average parallax factor.
        0 is good
        1 is bad
        """
        ave_ra, sumOfWeights = np.average(ra_pi_amp, weights=weights)
        ave_dec, sumOfWeights = np.average(dec_pi_amp, weights=weights)
        ave_r = np.sqrt( ave_ra**2 + ave_dec**2)

        return ave_r

    def computeWeights(self, dataSlice):
        filters = np.unique(dataSlice[self.filterCol])
        snr = np.zeros(len(dataSlice), dtype='float')
        # compute SNR for all observations
        for filt in filters:
            good = np.where(dataSlice[self.filterCol] == filt)
            snr[good] = m52snr(self.mags[filt], dataSlice[self.m5Col][good])
        # Compute centroid uncertainty in each visit
        position_errors = np.sqrt(astrom_precision(dataSlice[self.seeingCol], snr)**2+self.atm_err**2)
        weights = 1./position_errors**2
        return weights

    def run(self, dataSlice, slicePoint=None):

        # XXX-Need to add a SNR cut or something, so only include points that are relevant

        weights = self.computeWeights(dataSlice)
        aveFac = self._aveFactor(dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp'], weights)
        avePos = self._avePosition(dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp'])

        result = (1-avePos)*aveFac
        return result


class ParallaxHADegenMetric(BaseMetric):
    """
    Check for degeneracy between parallax and DCR
    """
    def __init__(self, metricName='ParallaxHADegenMetric',haCol='HA',**kwargs ):
        cols = ['ra_pi_amp', 'dec_pi_amp']
        self.haCol = haCol
        cols.append(haCol)
        units = 'Correlation'
        super(ParallaxHADegenMetric, self).__init__(cols,
                                                    metricName=metricName,
                                                    units=units, **kwargs)

        # XXX--need to update to include a SNR limit.

    def run(self, dataSlice, slicePoint=None):

        # Compute total parallax distance
        pf = np.sqrt(dataSlice['ra_pi_amp']**2+dataSlice['dec_pi_amp']**2)
        # Correlation between parallax factor and hour angle
        rho,p = spearmanr(pf, dataSlice[self.haCol])
        return rho
