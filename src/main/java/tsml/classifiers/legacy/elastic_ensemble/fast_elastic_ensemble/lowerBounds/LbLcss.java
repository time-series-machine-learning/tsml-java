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
 * A class to compute the lower bound function for LCSS distance
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LbLcss {
    /**
     * Build the upper and lower envelope for Lb Keogh with modification for LCSS
     *
     * @param candidate candidate sequence
     * @param epsilon   epsilon value
     * @param delta     delta value
     * @param U         upper envelope
     * @param L         lower envelope
     */
    public static void fillUL(final Instance candidate, final double epsilon, final int delta, final double[] U, final double[] L) {
        final int length = candidate.numAttributes() - 1;
        double min, max;

        for (int i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            final int startR = (i - delta < 0) ? 0 : i - delta;
            final int stopR = (i + delta + 1 > length) ? length : i + delta + 1;
            for (int j = startR; j < stopR; j++) {
                final double value = candidate.value(j);
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            L[i] = min - epsilon;
            U[i] = max + epsilon;
        }
    }

    /**
     * Lower bound for LCSS distance
     *
     * @param c candidate sequence
     * @param U upper envelope
     * @param L lower envelope
     * @return lower bound distance
     */
    public static double distance(final Instance c, final double[] U, final double[] L) {
        final int length = Math.min(U.length, c.numAttributes() - 1);

        double lcs = 0;

        for (int i = 0; i < length; i++) {
            if (c.value(i) <= U[i] && c.value(i) >= L[i]) {
                lcs++;
            }
        }

        return 1 - lcs / length;
    }

    /**
     * Lower bound for LCSS distance with early abandoning
     *
     * @param c           candidate sequence
     * @param U           upper envelope
     * @param L           lower envelope
     * @param cutOffValue cutoff value for early abandoning
     * @return lower bound distance
     */
    public static double distance(final Instance c, final double[] U, final double[] L, final double cutOffValue) {
        final int length = Math.min(U.length, c.numAttributes() - 1);
        final double ub = (1.0 - cutOffValue) * length;

        double lcs = 0;

        for (int i = 0; i < length; i++) {
            if (c.value(i) <= U[i] && c.value(i) >= L[i]) {
                lcs++;
                if (lcs <= ub) return 1;
            }
        }

        return 1 - lcs / length;
    }
}
