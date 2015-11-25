import numpy as np
from lsst.sims.maf.metrics import BaseMetric
from scipy.stats import spearmanr
from lsst.sims.maf.utils.astrometryUtils import m52snr, astrom_precision
import lsst.sims.maf.utils as utils

__all__ =['ParallaxCoverageMetric','ParallaxHADegenMetric']

class ParallaxCoverageMetric(BaseMetric):
    """
    Check how well the parallax factor is distributed. Multiplies the parallax factor (between 0 and 1-ish)
    with one minus the (weighted) average radius of the offset.
    """
    def __init__(self, metricName='ParallaxCoverageMetric', m5Col='fiveSigmaDepth',
                 mjdCol='expMJD', filterCol='filter', seeingCol='finSeeing',
                 rmag=20., SedTemplate='flat', badval=-666,
                 atm_err=0.01, thetaRange=np.pi/2., snrLimit=5, **kwargs):

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

    def _aveFactor(self, ra_pi_amp, dec_pi_amp, weights):
        """
        Find the average parallax factor:
        1 is good, 0 is bad
        """
        radius = np.sqrt(ra_pi_amp**2 + dec_pi_amp**2 )
        # sigma_r = sigma_ra = sigma_dec
        return np.average(radius, weights=weights)

    def _avePosition(self, ra_pi_amp, dec_pi_amp, weights):
        """
        Find the average parallax factor.
        0 is good
        1 is bad
        """
        ave_ra = np.average(ra_pi_amp, weights=weights)
        ave_dec = np.average(dec_pi_amp, weights=weights)
        ave_r = np.sqrt( ave_ra**2 + ave_dec**2)
        return ave_r

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


    def weightedAngle(self, dec_pi_amp, ra_pi_amp, weights):
        """ https://en.wikipedia.org/wiki/Mean_of_circular_quantities"""
        thetaMean = np.arctan2(np.average(dec_pi_amp, weights=weights),
                               np.average(ra_pi_amp, weights=weights))

    # OK! If I subtract off the weighted mean position, then compute the mean radius of the points!

    def weightedR(self, dec_pi_amp, ra_pi_amp, weights):
        ycoord = dec_pi_amp-np.average(dec_pi_amp, weights=weights)
        xcoord = ra_pi_amp-np.average(ra_pi_amp,weights=weights)
        radius = np.sqrt(xcoord**2+ycoord**2)
        aveRad = np.average(radius, weights=weights)
        return aveRad


    def thetaWeight(self, dec_pi_amp, ra_pi_amp, weights):
        # We want theta to average to zero. And then theta+90 degrees to be zero

        theta = np.arctan2(dec_pi_amp, ra_pi_amp)
        theta[np.where(theta < 0)] = theta[np.where(theta < 0)]+2.*np.pi
        theta = theta-np.min(theta)
        theta = theta % 2.*np.pi
        # weight between 0-1
        thetaWeight1 = np.average(np.abs(theta), weights=weights)/np.pi

        thetaWeight2 = np.average(np.abs(theta2), weights=weights)/np.pi




    def radiusWeight(self, dec_pi_amp, ra_pi_amp, weights):
        radius = np.sqrt(ave_ra**2 + ave_dec**2)
        radiusWeighted = np.average(radius, weights=weights)



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
        aveFac = self._aveFactor(dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp'], weights)
        avePos = self._avePosition(dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp'], weights)
        thetaCheck = self._thetaCheck(dataSlice['ra_pi_amp'], dataSlice['dec_pi_amp'], snr)
        result = (1-avePos)*aveFac*thetaCheck
        return result


class ParallaxHADegenMetric(BaseMetric):
    """
    Check for degeneracy between parallax and DCR
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
