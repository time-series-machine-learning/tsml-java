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
 * A class to compute the lower bounds for the ERP distance
 * Lower bound function for ERP distance
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LbErp {
    private static double[] s = new double[2];

    /**
     * Proposed lower bound function for ERP
     * |sum(Q)-sum(C)|
     * Modified slightly to have g value
     * http://www.vldb.org/conf/2004/RS21P2.PDF
     * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.6387&rep=rep1&type=pdf
     *
     * @param a first time series
     * @param b second time series
     * @param g g parameter
     * @return LB ERP distance distance
     */
    public static double distance(final Instance a, final Instance b, final double g) {
        final int m = a.numAttributes() - 1;
        final int n = b.numAttributes() - 1;

        if (m == n) {
            sum2(a, b, g);
            return Math.abs(s[0] - s[1]);
        } else {
            return Math.abs(sum(a, g) - sum(b, g));
        }
    }

    /**
     * Sum of all points in one sequence
     *
     * @param a time series
     * @param g g parameter value
     * @return sum
     */
    private static double sum(final Instance a, final double g) {
        double s = 0;
        for (int i = 0; i < a.numAttributes() - 1; i++) {
            s += Math.abs(a.value(i) - g);
        }

        return s;
    }

    /**
     * Sum of all points in 2 sequences
     *
     * @param a first time series
     * @param b second time series
     * @param g g parameter value
     */
    private static void sum2(final Instance a, final Instance b, final double g) {
        s = new double[2];
        for (int i = 0; i < a.numAttributes() - 1; i++) {
            s[0] += Math.abs(a.value(i) - g);
            s[1] += Math.abs(b.value(i) - g);
        }
    }

    /**
     * Build the upper and lower envelope for LB Keogh with modification for ERP
     *
     * @param a        time series
     * @param g        g parameter value
     * @param bandSize size of the warping window
     * @param U        upper envelope
     * @param L        lower envelope
     */
    public static void fillUL(final Instance a, final double g, final double bandSize, final double[] U, final double[] L) {
        final int length = a.numAttributes() - 1;
        final int r = (int) Math.ceil(length * bandSize);
        double min, max;

        for (int i = 0; i < length; i++) {
            min = g;
            max = g;
            final int startR = Math.max(0, i - r);
            final int stopR = Math.min(length - 1, i + r);
            for (int j = startR; j <= stopR; j++) {
                final double value = a.value(j);
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            U[i] = max;
            L[i] = min;
        }
    }

    /**
     * LB Keogh version for ERP
     *
     * @param a time series
     * @param U upper envelope
     * @param L lower envelope
     * @return LB Keogh distance for ERP
     */
    public static double distance(final Instance a, final double[] U, final double[] L) {
        return LbKeogh.distance(a, U, L);
    }

    /**
     * LB Keogh version for ERP with early abandon
     *
     * @param a           time series
     * @param U           upper envelope
     * @param L           lower envelope
     * @param cutOffValue cutoff value for early abandon
     * @return LB Keogh distance for ERP
     */
    public static double distance(final Instance a, final double[] U, final double[] L, final double cutOffValue) {
        return LbKeogh.distance(a, U, L, cutOffValue);
    }
}
