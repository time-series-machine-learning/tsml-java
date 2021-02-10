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

import tsml.classifiers.legacy.elastic_ensemble.distance_functions.DTW;
import tsml.transformers.Derivative;
import weka.core.Instance;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * A class to compute the LB Keogh for DTW distance
 * @article{keogh2005exact,
 *   title={Exact indexing of dynamic time warping},
 *   author={Keogh, Eamonn and Ratanamahatana, Chotirat Ann},
 *   journal={Knowledge and information systems},
 *   volume={7},
 *   number={3},
 *   pages={358--386},
 *   year={2005},
 *   publisher={Springer}
 * }
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LbKeogh {
    private static Derivative df = new Derivative();

    public static void fillUL(final double[] sequence, final int window, final double[] U, final double[] L) {
        final int length = sequence.length;
        double min, max;

        for (int i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            final int startR = Math.max(0, i - window);
            final int stopR = Math.min(length - 1, i + window);
            for (int j = startR; j <= stopR; j++) {
                final double value = sequence[j];
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            L[i] = min;
            U[i] = max;
        }
    }

    public static void fillUL(final Instance sequence, final int window, final double[] U, final double[] L) {
        final int length = sequence.numAttributes() - 1;
        double min, max;

        for (int i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            final int startR = Math.max(0, i - window);
            final int stopR = Math.min(length - 1, i + window);
            for (int j = startR; j <= stopR; j++) {
                final double value = sequence.value(j);
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            L[i] = min;
            U[i] = max;
        }
    }


    public static void fillUL(final Instance sequence, final double r, final double[] U, final double[] L) {
        final int length = sequence.numAttributes();
        double min, max;
        final int window = DTW.getWindowSize(length, r);

        for (int i = 0; i < length; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            final int startR = Math.max(0, i - window);
            final int stopR = Math.min(length - 1, i + window);
            for (int j = startR; j <= stopR; j++) {
                final double value = sequence.value(j);
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            L[i] = min;
            U[i] = max;
        }
    }

    public static void fillULStreaming(final double[] y, final int r, final double[] U, final double[] L) {
        Deque<Integer> u = new ArrayDeque<>();
        Deque<Integer> l = new ArrayDeque<>();
        u.addLast(0);
        l.addLast(0);
        final int width = 1 + 2 * r;
        int i;
        for (i = 1; i < y.length; ++i) {
            if (i >= r + 1) {
                U[i - r - 1] = y[u.getFirst()];
                L[i - r - 1] = y[l.getFirst()];
            }
            if (y[i] > y[i - 1]) {
                u.removeLast();
                while (u.size() > 0) {
                    if (y[i] <= y[u.getLast()]) break;
                    u.removeLast();
                }
            } else {
                l.removeLast();
                while (l.size() > 0) {
                    if (y[i] >= y[l.getLast()]) break;
                    l.removeLast();
                }
            }
            u.addLast(i);
            l.addLast(i);
            if (i == width + u.getFirst()) {
                u.removeFirst();
            } else if (i == width + l.getFirst()) {
                l.removeFirst();
            }
        }

        for (i = y.length; i <= y.length + r; ++i) {
            final int index = Math.max(i - r - 1, 0);
            U[index] = y[u.getFirst()];
            L[index] = y[l.getFirst()];
            if (i - u.getFirst() >= width) {
                u.removeFirst();
            }
            if (i - l.getFirst() >= width) {
                l.removeFirst();
            }
        }
    }

    public static double distance(final Instance a, final double[] U, final double[] L) {
        final int length = Math.min(U.length, a.numAttributes() - 1);
        double res = 0;

        for (int i = 0; i < length; i++) {
            final double c = a.value(i);
            if (c < L[i]) {
                final double diff = L[i] - c;
                res += diff * diff;
            } else if (U[i] < c) {
                final double diff = U[i] - c;
                res += diff * diff;
            }
        }

        return res;
    }

    public static double distance(final Instance a, final double[] U, final double[] L, final double cutOffValue) {
        final int length = Math.min(U.length, a.numAttributes() - 1);
        double res = 0;

        for (int i = 0; i < length; i++) {
            final double c = a.value(i);
            if (c < L[i]) {
                final double diff = L[i] - c;
                res += diff * diff;
                if (res >= cutOffValue)
                    return Double.MAX_VALUE;
            } else if (U[i] < c) {
                final double diff = U[i] - c;
                res += diff * diff;
                if (res >= cutOffValue)
                    return Double.MAX_VALUE;
            }
        }

        return res;
    }
}
