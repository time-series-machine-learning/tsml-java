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
package tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils;

/**
 * This class contains some generic tools
 *
 * @author Chang Wei Tan (chang.tan@monash.edu)
 */
public class GenericTools {
    public static double distanceTo(final double a, final double b) {
        double diff = a - b;
        return diff * diff;
    }

    public static int argMin3(final double a, final double b, final double c) {
        return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
    }

    public static int argMax3(final double a, final double b, final double c) {
        return (a >= b) ? ((a >= c) ? 0 : 2) : (b >= c) ? 1 : 2;
    }

    public static double min3(final double a, final double b, final double c) {
        return (a <= b) ? ((a <= c) ? a : c) : (b <= c) ? b : c;
    }

    public static double max3(final double a, final double b, final double c) {
        return (a >= b) ? ((a >= c) ? a : c) : (b >= c) ? b : c;
    }

    public static double[] minmax(final double[] a) {
        final int n = a.length;
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        for (double v : a) {
            if (v > max) max = v;
            if (v < min) min = v;
        }

        return new double[]{min, max};
    }

    public static int[] getInclusive10(final int min, final int max) {
        int[] output = new int[10];

        double diff = 1.0 * (max - min) / 9;
        double[] doubleOut = new double[10];
        doubleOut[0] = min;
        output[0] = min;
        for (int i = 1; i < 9; i++) {
            doubleOut[i] = doubleOut[i - 1] + diff;
            output[i] = (int) Math.round(doubleOut[i]);
        }
        output[9] = max; // to make sure max isn't omitted due to double imprecision
        return output;
    }

    public static double[] getInclusive10(final double min, final double max) {
        double[] output = new double[10];
        double diff = 1.0 * (max - min) / 9;
        output[0] = min;
        for (int i = 1; i < 9; i++) {
            output[i] = output[i - 1] + diff;
        }
        output[9] = max;

        return output;
    }
}
