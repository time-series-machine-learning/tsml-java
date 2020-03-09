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
 * A class to compute the lower bound for TWE distance.
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LbTwed {
    /**
     * Lower bound for TWED distance
     *
     * @param q      first time series
     * @param c      second time series
     * @param qMax   max of the first time series
     * @param qMin   min of the second time series
     * @param nu     stiffness parameter
     * @param lambda constant penalty
     * @return lower bound distance
     */
    public static double distance(final Instance q, final Instance c, final double qMax, final double qMin,
                                  final double nu, final double lambda) {
        final int length = q.numAttributes() - 1;
        final double q0 = q.value(0);
        final double c0 = c.value(0);
        double diff = q0 - c0;
        double res = Math.min(diff * diff,
                Math.min(q0 * q0 + nu + lambda,
                        c0 * c0 + nu + lambda));

        for (int i = 1; i < length; i++) {
            final double curr = c.value(i);
            final double prev = c.value(i - 1);
            final double max = Math.max(qMax, prev);
            final double min = Math.min(qMin, prev);
            if (curr < min) {
                diff = min - curr;
                res += Math.min(nu, diff * diff);
            } else if (max < curr) {
                diff = max - curr;
                res += Math.min(nu, diff * diff);
            }
        }

        return res;
    }

    /**
     * Lower bound for TWED distance with early abandon
     *
     * @param q           first time series
     * @param c           second time series
     * @param qMax        max of the first time series
     * @param qMin        min of the second time series
     * @param nu          stiffness parameter
     * @param lambda      constant penalty
     * @param cutOffValue cutoff value for early abandon
     * @return lower bound distance
     */
    public static double distance(final Instance q, final Instance c, final double qMax, final double qMin,
                                  final double nu, final double lambda, final double cutOffValue) {
        final int length = q.numAttributes() - 1;
        final double q0 = q.value(0);
        final double c0 = c.value(0);
        double diff = q0 - c0;
        double res = Math.min(diff * diff,
                Math.min(q0 * q0 + nu + lambda,
                        c0 * c0 + nu + lambda));
        if (res >= cutOffValue)
            return res;

        for (int i = 1; i < length; i++) {
            final double curr = c.value(i);
            final double prev = c.value(i - 1);
            final double max = Math.max(qMax, prev);
            final double min = Math.min(qMin, prev);
            if (curr < min) {
                diff = min - curr;
                res += Math.min(nu, diff * diff);
                if (res >= cutOffValue)
                    return res;
            } else if (max < curr) {
                diff = max - curr;
                res += Math.min(nu, diff * diff);
                if (res >= cutOffValue)
                    return res;
            }
        }

        return res;
    }
}
