import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from scipy.stats import spearmanr

__all__ =['ParallaxCoverageMetric','ParallaxHADegenMetric']

class ParallaxCoverageMetric(BaseMetric):
    """
    Check how well the parallax factor is distributed.
    """
    def __init__(self, metricName='ParallaxCoverageMetric', **kwargs):
        cols = ['ra_pi_amp', 'dec_pi_amp']
        units = 'ratio'
        super(ParallaxCoverageMetric, self).__init__(cols,
                                                     metricName=metricName, units=units,
                                                      **kwargs)

    def _aveFactor(self, ra_pi_amp, dec_pi_amp):
        """
        Find the average parallax factor:
        1 is good, 0 is bad
        """
        return np.mean(np.sqrt(ra_pi_amp**2 + dec_pi_amp**2 ))

    def _avePosition(self, ra_pi_amp, dec_pi_amp):
        """
        Find the average parallax factor.
        0 is good
        1 is bad
        """
        ave_ra = np.mean(ra_pi_amp)
        ave_dec = np.mean(dec_pi_amp)
        ave_r = np.sqrt( ave_ra**2 + ave_dec**2)

        return ave_r

    def run(self, dataSlice, slicePoint=None):

        # XXX-Need to add a SNR cut or something, so only include points that are relevant

        aveFac = self._aveFactor(dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp'])
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
        # I can take a dot product of parallax displacement and the DCR displacement.

    def run(self, dataSlice, slicePoint=None):

        # Compute total parallax distance
        pf = np.sqrt(dataSlice['ra_pi_amp']**2+dataSlice['dec_pi_amp']**2)
        # Correlation between parallax factor and hour angle
        rho,p = spearmanr(pf, dataSlice[self.haCol])
        return rho
