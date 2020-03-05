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
 * A class to compute Lb Improved for DTW distance
 * @article{lemire2009faster,
 *   title={Faster retrieval with a two-pass dynamic-time-warping lower bound},
 *   author={Lemire, Daniel},
 *   journal={Pattern recognition},
 *   volume={42},
 *   number={9},
 *   pages={2169--2180},
 *   year={2009},
 *   publisher={Elsevier}
 * }
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class LbImproved {
    public static double distance(final Instance a, final Instance b, final double[] Ub, final double[] Lb, final int r) {
        final int length = Math.min(Ub.length, a.numAttributes() - 1);
        final double[] y = new double[length];
        final double[] Ux = new double[length];
        final double[] Lx = new double[length];

        int i;
        double res = 0;
        double diff, c;

        for (i = 0; i < length; i++) {
            c = a.value(i);
            if (c < Lb[i]) {
                diff = Lb[i] - c;
                res += diff * diff;
                y[i] = Lb[i];
            } else if (Ub[i] < c) {
                diff = c - Ub[i];
                res += diff * diff;
                y[i] = Ub[i];
            } else {
                y[i] = c;
            }
        }

        LbKeogh.fillULStreaming(y, r, Ux, Lx);
        for (i = 0; i < length; i++) {
            c = b.value(i);
            if (c < Lx[i]) {
                diff = Lx[i] - c;
                res += diff * diff;
            } else if (Ux[i] < c) {
                diff = c - Ux[i];
                res += diff * diff;
            }
        }

        return res;
    }

    public double distance(final Instance a, final Instance b, final double[] Ub, final double[] Lb, final int r, final double cutOffValue) {
        final int length = Math.min(Ub.length, a.numAttributes() - 1);
        final double[] y = new double[length];
        int i;
        double res = 0;
        double diff, c;

        for (i = 0; i < length; i++) {
            c = a.value(i);
            if (c < Lb[i]) {
                diff = Lb[i] - c;
                res += diff * diff;
                y[i] = Lb[i];
            } else if (Ub[i] < c) {
                diff = c - Ub[i];
                res += diff * diff;
                y[i] = Ub[i];
            } else {
                y[i] = c;
            }
        }
        if (res < cutOffValue) {
            final double[] Ux = new double[length];
            final double[] Lx = new double[length];
            LbKeogh.fillULStreaming(y, r, Ux, Lx);
            for (i = 0; i < length; i++) {
                c = b.value(i);
                if (c < Lx[i]) {
                    diff = Lx[i] - c;
                    res += diff * diff;
                } else if (Ux[i] < c) {
                    diff = c - Ux[i];
                    res += diff * diff;
                }
            }
        }
        return res;
    }

}
