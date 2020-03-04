/* Copyright (C) 2019 Chang Wei Tan
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. */
package tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds;

import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import weka.core.Instance;

/**
 * A class to compute the lower bounds for MSM distance
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LbMsm {
    /**
     * Lower bound distance for MSM
     *
     * @param q    first time series
     * @param c    second time series
     * @param cc   c param
     * @param qMax max of first time series
     * @param qMin min of second time series
     * @return lower bound distance
     */
    public static double distance(final Instance q, final Instance c, final double cc, final double qMax, final double qMin) {
        final int len = q.numAttributes() - 1;

        double d = Math.abs(q.value(0) - c.value(0));

        for (int i = 1; i < len; i++) {
            final double curr = c.value(i);
            final double prev = c.value(i - 1);
            if (prev >= curr && curr > qMax) {
                d += Math.min(Math.abs(curr - qMax), cc);
            } else if (prev <= curr && curr < qMin) {
                d += Math.min(Math.abs(curr - qMin), cc);
            }
        }

        return d;
    }

    /**
     * Lower bound distance for MSM with early abandon
     *
     * @param q           first time series
     * @param c           second time series
     * @param cc          c param
     * @param qMax        max of first time series
     * @param qMin        min of second time series
     * @param cutOffValue cutoff value for early abandon
     * @return lower bound distance
     */
    public static double distance(final Instance q, final Instance c, final double cc, final double qMax, final double qMin, final double cutOffValue) {
        final int len = q.numAttributes() - 1;

        double d = Math.abs(q.value(0) - c.value(0));

        for (int i = 1; i < len; i++) {
            final double curr = c.value(i);
            final double prev = c.value(i - 1);
            if (prev >= curr && curr > qMax) {
                d += Math.min(Math.abs(curr - qMax), cc);
                if (d >= cutOffValue)
                    return Double.MAX_VALUE;
            } else if (prev <= curr && curr < qMin) {
                d += Math.min(Math.abs(curr - qMin), cc);
                if (d >= cutOffValue)
                    return Double.MAX_VALUE;
            }
        }

        return d;
    }

    /**
     * Lb Kim version for LB MSM
     *
     * @param q              query sequence
     * @param c              reference sequence
     * @param queryCache     cache for query
     * @param candidateCache cache for reference
     * @param indexQuery     query index
     * @param indexCandidate reference index
     * @return lower bound distance
     */
    public static double distance(final Instance q, final Instance c,
                                  final SequenceStatsCache queryCache, final SequenceStatsCache candidateCache,
                                  final int indexQuery, final int indexCandidate) {
        final double diffFirsts = Math.abs(q.value(0) - c.value(0));
        final double diffLasts = Math.abs(q.value(q.numAttributes() - 2) - c.value(c.numAttributes() - 2));
        double minDist = diffFirsts + diffLasts;

        if (!queryCache.isMinFirst(indexQuery) && !candidateCache.isMinFirst(indexCandidate) &&
                !queryCache.isMinLast(indexQuery) && !candidateCache.isMinLast(indexCandidate)) {
            minDist += Math.abs(queryCache.getMin(indexQuery) - candidateCache.getMin(indexCandidate));
        }
        if (!queryCache.isMaxFirst(indexQuery) && !candidateCache.isMaxFirst(indexCandidate) &&
                !queryCache.isMaxLast(indexQuery) && !candidateCache.isMaxLast(indexCandidate)) {
            minDist += Math.abs(queryCache.getMax(indexQuery) - candidateCache.getMax(indexCandidate));
        }

        return minDist;
    }
}
