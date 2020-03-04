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

import weka.core.Instance;

/**
 * A class to compute the lower bounds for WDTW distance
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LbWdtw {
    /**
     * WDTW lower bound using envelopes
     *
     * @param candidate time series
     * @param weight    minimum weight
     * @param queryMax  maximum of the other time series
     * @param queryMin  minimum of the other time series
     * @return lower bound distance
     */
    public static double distance(final Instance candidate, final double weight, final double queryMax, final double queryMin) {
        double res = 0;

        for (int i = 0; i < candidate.numAttributes() - 1; i++) {
            final double c = candidate.value(i);
            if (c < queryMin) {
                final double diff = queryMin - c;
                res += diff * diff;
            } else if (queryMax < c) {
                final double diff = queryMax - c;
                res += diff * diff;
            }
        }

        return weight * res;
    }


    /**
     * WDTW lower bound using envelopes and early abandon
     *
     * @param candidate   time series
     * @param weight      minimum weight
     * @param queryMax    maximum of the other time series
     * @param queryMin    minimum of the other time series
     * @param cutOffValue cutoff value for early abandon
     * @return lower bound distance
     */
    public static double distance(final Instance candidate, final double weight, final double queryMax, final double queryMin, final double cutOffValue) {
        double res = 0;
        double cutoff = cutOffValue / weight;

        for (int i = 0; i < candidate.numAttributes() - 1; i++) {
            final double c = candidate.value(i);
            if (c < queryMin) {
                final double diff = queryMin - c;
                res += diff * diff;
                if (res >= cutoff)
                    return Double.MAX_VALUE;
            } else if (queryMax < c) {
                final double diff = queryMax - c;
                res += diff * diff;
                if (res >= cutoff)
                    return Double.MAX_VALUE;
            }
        }
        return weight * res;
    }
}
