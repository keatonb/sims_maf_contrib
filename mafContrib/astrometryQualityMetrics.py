import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from scipy.stats import spearmanr
from lsst.sims.maf.utils.astrometryUtils import m52snr, astrom_precision
import lsst.sims.maf.utils as utils

__all__ =['ParallaxCoverageMetric','ParallaxHADegenMetric']

class ParallaxCoverageMetric(BaseMetric):
    """
    Check how well the parallax factor is distributed. subtracts the weighted mean position of the
    parallax offsets, then computes the weighted mean radius of the points.
    If points are well distributed, the radius will be near 1. If phase coverage is bad,
    radius will be close to zero.

    For points on the Ecliptic, uniform sampling should result in a metric value of ~0.5.
    At the poles, uniform sampling would result in a metric value of ~1.

    Also demand that there are obsevations above the snrLimit kwarg spanning thetaRange radians.
    """
    def __init__(self, metricName='ParallaxCoverageMetric', m5Col='fiveSigmaDepth',
                 mjdCol='expMJD', filterCol='filter', seeingCol='finSeeing',
                 rmag=20., SedTemplate='flat', badval=-666,
                 atm_err=0.01, thetaRange=0., snrLimit=5, **kwargs):

        cols = ['ra_pi_amp', 'dec_pi_amp', m5Col, mjdCol, filterCol, seeingCol]
        units = 'ratio'
        super(ParallaxCoverageMetric, self).__init__(cols,
                                                     metricName=metricName, units=units,
                                                      **kwargs)
        self.m5Col = m5Col
        self.seeingCol = seeingCol
        self.filterCol = filterCol
        self.mjdCol = mjdCol

        # Demand the range of theta values
        self.thetaRange = thetaRange
        self.snrLimit = snrLimit

        filters=['u','g','r','i','z','y']
        self.mags={}
        if SedTemplate == 'flat':
            for f in filters:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(SedTemplate, rmag=rmag)
        self.atm_err = atm_err


    def _thetaCheck(self, ra_pi_amp, dec_pi_amp, snr):
        good = np.where(snr >= self.snrLimit)
        theta = np.arctan2(dec_pi_amp[good], ra_pi_amp[good])
        # Make values between 0 and 2pi
        theta = theta-np.min(theta)
        result = 0.
        if np.max(theta) >= self.thetaRange:
            # Check that things are in differnet quadrants
            theta = (theta+np.pi) % 2.*np.pi
            theta = theta-np.min(theta)
            if np.max(theta) >= self.thetaRange:
                result = 1
        return result

    def computeWeights(self, dataSlice, snr):
        # Compute centroid uncertainty in each visit
        position_errors = np.sqrt(astrom_precision(dataSlice[self.seeingCol], snr)**2+self.atm_err**2)
        weights = 1./position_errors**2
        return weights


    def weightedR(self, dec_pi_amp, ra_pi_amp, weights):
        ycoord = dec_pi_amp-np.average(dec_pi_amp, weights=weights)
        xcoord = ra_pi_amp-np.average(ra_pi_amp,weights=weights)
        radius = np.sqrt(xcoord**2+ycoord**2)
        aveRad = np.average(radius, weights=weights)
        return aveRad


    def run(self, dataSlice, slicePoint=None):

        if np.size(dataSlice) < 2:
            return self.badval

        filters = np.unique(dataSlice[self.filterCol])
        snr = np.zeros(len(dataSlice), dtype='float')
        # compute SNR for all observations
        for filt in filters:
            good = np.where(dataSlice[self.filterCol] == filt)
            snr[good] = m52snr(self.mags[filt], dataSlice[self.m5Col][good])

        weights = self.computeWeights(dataSlice, snr)
        aveR = self.weightedR(dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp'], weights)
        if self.thetaRange > 0:
            thetaCheck = self._thetaCheck(dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp'], snr)
        else:
            thetaCheck = 1.
        result = aveR*thetaCheck
        return result


class ParallaxHADegenMetric(BaseMetric):
    """
    Check for degeneracy between parallax and DCR.  Value of zero means there is no correlation.
    Values of +/-1 mean correlation (or anti-correlation, which is probably just as bad).
    """
    def __init__(self, metricName='ParallaxHADegenMetric',haCol='HA', snrLimit=5.,
                 m5Col='fiveSigmaDepth', mjdCol='expMJD',
                 filterCol='filter', seeingCol='finSeeing',
                 rmag=20., SedTemplate='flat', badval=-666,
                 atm_err=0.01,**kwargs ):
        cols = ['ra_pi_amp', 'dec_pi_amp']
        self.haCol = haCol
        cols.append(haCol)
        units = 'Correlation'
        self.snrLimit = snrLimit
        super(ParallaxHADegenMetric, self).__init__(cols,
                                                    metricName=metricName,
                                                    units=units, **kwargs)
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


    def run(self, dataSlice, slicePoint=None):

        if np.size(dataSlice) < 2:
            return self.badval
        filters = np.unique(dataSlice[self.filterCol])
        snr = np.zeros(len(dataSlice), dtype='float')
        # compute SNR for all observations
        for filt in filters:
            good = np.where(dataSlice[self.filterCol] == filt)
            snr[good] = m52snr(self.mags[filt], dataSlice[self.m5Col][good])
        # Compute total parallax distance
        pf = np.sqrt(dataSlice['ra_pi_amp']**2+dataSlice['dec_pi_amp']**2)
        # Correlation between parallax factor and hour angle
        aboveLimit = np.where(snr >= self.snrLimit)[0]
        if np.size(aboveLimit) < 2:
            return self.badval
        rho,p = spearmanr(pf[aboveLimit], dataSlice[self.haCol][aboveLimit])
        return rho
