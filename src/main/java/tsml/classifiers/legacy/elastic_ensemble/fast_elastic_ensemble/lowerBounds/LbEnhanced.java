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
 * A class to compute the Enhanced lower bound for DTW distance
 * @inproceedings{tan2019elastic,
 *   title={Elastic bands across the path: A new framework and method to lower bound DTW},
 *   author={Tan, Chang Wei and Petitjean, Fran{\c{c}}ois and Webb, Geoffrey I},
 *   booktitle={Proceedings of the 2019 SIAM International Conference on Data Mining},
 *   pages={522--530},
 *   year={2019},
 *   organization={SIAM}
 * }
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */

public class LbEnhanced {
    public static double distance(final Instance a, final Instance b,
                                  final double[] U, final double[] L,
                                  final int w, final int v, final double cutOffValue) {
        final int n = a.numAttributes() - 1;
        final int m = b.numAttributes() - 1;
        final int l = n - 1;
        final int nBands = Math.min(l / 2, v);
        final int lastIndex = l - nBands;
        final double d00 = a.value(0) - b.value(0);
        final double dnm = a.value(n - 1) - b.value(m - 1);

        int i, j, rightEnd, rightStart;
        double minL, minR, tmp, aVal;

        double res = d00 * d00 + dnm * dnm;

        for (i = 1; i < nBands; i++) {
            rightEnd = l - i;
            minL = a.value(i) - b.value(i);
            minL *= minL;
            minR = a.value(rightEnd) - b.value(rightEnd);
            minR *= minR;
            for (j = Math.max(0, i - w); j < i; j++) {
                rightStart = l - j;
                tmp = a.value(i) - b.value(j);
                minL = Math.min(minL, tmp * tmp);
                tmp = a.value(j) - b.value(i);
                minL = Math.min(minL, tmp * tmp);

                tmp = a.value(rightEnd) - b.value(rightStart);
                minR = Math.min(minR, tmp * tmp);
                tmp = a.value(rightStart) - b.value(rightEnd);
                minR = Math.min(minR, tmp * tmp);
            }
            res += minL + minR;
        }
        if (res >= cutOffValue)
            return Double.POSITIVE_INFINITY;

        for (i = nBands; i <= lastIndex; i++) {
            aVal = a.value(i);
            if (aVal > U[i]) {
                tmp = aVal - U[i];
                res += tmp * tmp;
            } else if (aVal < L[i]) {
                tmp = L[i] - aVal;
                res += tmp * tmp;
            }
        }

        return res;
    }

    public static double distance(final Instance a, final Instance b,
                                  final double[] U, final double[] L,
                                  final int w, final int v) {
        final int n = a.numAttributes() - 1;
        final int m = b.numAttributes() - 1;
        final int l = n - 1;
        final int nBands = Math.min(l / 2, v);
        final int lastIndex = l - nBands;
        final double d00 = a.value(0) - b.value(0);
        final double dnm = a.value(n - 1) - b.value(m - 1);

        int i, j, rightEnd, rightStart;
        double minL, minR, tmp, aVal;

        double res = d00 * d00 + dnm * dnm;

        for (i = 1; i < nBands; i++) {
            rightEnd = l - i;
            minL = a.value(i) - b.value(i);
            minL *= minL;
            minR = a.value(rightEnd) - b.value(rightEnd);
            minR *= minR;
            for (j = Math.max(0, i - w); j < i; j++) {
                rightStart = l - j;
                tmp = a.value(i) - b.value(j);
                minL = Math.min(minL, tmp * tmp);
                tmp = a.value(j) - b.value(i);
                minL = Math.min(minL, tmp * tmp);

                tmp = a.value(rightEnd) - b.value(rightStart);
                minR = Math.min(minR, tmp * tmp);
                tmp = a.value(rightStart) - b.value(rightEnd);
                minR = Math.min(minR, tmp * tmp);
            }
            res += minL + minR;
        }

        for (i = nBands; i <= lastIndex; i++) {
            aVal = a.value(i);
            if (aVal > U[i]) {
                tmp = aVal - U[i];
                res += tmp * tmp;
            } else if (aVal < L[i]) {
                tmp = L[i] - aVal;
                res += tmp * tmp;
            }
        }

        return res;
    }
}
