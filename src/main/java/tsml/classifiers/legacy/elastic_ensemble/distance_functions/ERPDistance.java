package tsml.classifiers.legacy.elastic_ensemble.distance_functions;

/*

This file is part of ELKI:
Environment for Developing KDD-Applications Supported by Index-Structures

Copyright (C) 2011
Ludwig-Maximilians-UniversitÃ¤t MÃ¼nchen
Lehr- und Forschungseinheit fÃ¼r Datenbanksysteme
ELKI Development Team

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Modified by Jason Lines (j.lines@uea.ac.uk)

 */

import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.WarpingPathResults;
import weka.core.Instance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;



public class ERPDistance extends EuclideanDistance {

    private double g;
    private double bandSize;

    public ERPDistance(double g, double bandSize) {
        this.g = g;
        this.bandSize = bandSize;
    }

    /**
     * Distance method
     *
     * @param first       instance 1
     * @param second      instance 2
     * @param cutOffValue used for early abandon
     * @param stats
     * @return distance between instances
     */
    @Override
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) {
        //Get the double arrays
        return distance(first, second, cutOffValue);
    }

    /**
     * distance method that converts instances to arrays of doubles
     *
     * @param first       instance 1
     * @param second      instance 2
     * @param cutOffValue used for early abandon
     * @return distance between instances
     */
    @Override
    public double distance(Instance first, Instance second, double cutOffValue) {
        //remove class index from first instance if there is one
        int firtClassIndex = first.classIndex();
        double[] arr1;
        if (firtClassIndex > 0) {
            arr1 = new double[first.numAttributes() - 1];
            for (int i = 0, j = 0; i < first.numAttributes(); i++) {
                if (i != firtClassIndex) {
                    arr1[j] = first.value(i);
                    j++;
                }
            }
        } else {
            arr1 = first.toDoubleArray();
        }

        //remove class index from second instance if there is one
        int secondClassIndex = second.classIndex();
        double[] arr2;
        if (secondClassIndex > 0) {
            arr2 = new double[second.numAttributes() - 1];
            for (int i = 0, j = 0; i < second.numAttributes(); i++) {
                if (i != secondClassIndex) {
                    arr2[j] = second.value(i);
                    j++;
                }
            }
        } else {
            arr2 = second.toDoubleArray();
        }

        return distance(arr1, arr2, cutOffValue);
    }

    public double distance(double[] first, double[] second, double cutOffValue) {
//        return ERPDistance(first, second);
        return ERPDistance(new NumberVector(first), new NumberVector(second));

    }


    private static class NumberVector {

        private double[] values;

        public NumberVector(double[] values) {
            this.values = values;
        }

        public int getDimensionality() {
            return values.length;
        }

        public double doubleValue(int dimension) {
            try {
                return values[dimension - 1];
            } catch (IndexOutOfBoundsException e) {
                throw new IllegalArgumentException("Dimension " + dimension + " out of range.");
            }
        }
    }


    //public double doubleDistance(NumberVector<?, ?> v1, NumberVector<?, ?> v2) {
    public double ERPDistance(NumberVector v1, NumberVector v2) {
        // Current and previous columns of the matrix
        double[] curr = new double[v2.getDimensionality()];
        double[] prev = new double[v2.getDimensionality()];

        // size of edit distance band
        // bandsize is the maximum allowed distance to the diagonal
//        int band = (int) Math.ceil(v2.getDimensionality() * bandSize);
        int band = (int) Math.ceil(v2.getDimensionality() * bandSize);

        // g parameter for local usage
        double gValue = g;

        for (int i = 0; i < v1.getDimensionality(); i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            {
                double[] temp = prev;
                prev = curr;
                curr = temp;
            }
            int l = i - (band + 1);
            if (l < 0) {
                l = 0;
            }
            int r = i + (band + 1);
            if (r > (v2.getDimensionality() - 1)) {
                r = (v2.getDimensionality() - 1);
            }

            for (int j = l; j <= r; j++) {
                if (Math.abs(i - j) <= band) {
                    // compute squared distance of feature vectors
                    double val1 = v1.doubleValue(i + 1);
                    double val2 = gValue;
                    double diff = (val1 - val2);
                    final double d1 = Math.sqrt(diff * diff);

                    val1 = gValue;
                    val2 = v2.doubleValue(j + 1);
                    diff = (val1 - val2);
                    final double d2 = Math.sqrt(diff * diff);

                    val1 = v1.doubleValue(i + 1);
                    val2 = v2.doubleValue(j + 1);
                    diff = (val1 - val2);
                    final double d12 = Math.sqrt(diff * diff);

                    final double dist1 = d1 * d1;
                    final double dist2 = d2 * d2;
                    final double dist12 = d12 * d12;

                    final double cost;

                    if ((i + j) != 0) {
                        if ((i == 0) || ((j != 0) && (((prev[j - 1] + dist12) > (curr[j - 1] + dist2)) && ((curr[j - 1] + dist2) < (prev[j] + dist1))))) {
                            // del
                            cost = curr[j - 1] + dist2;
                        } else if ((j == 0) || ((i != 0) && (((prev[j - 1] + dist12) > (prev[j] + dist1)) && ((prev[j] + dist1) < (curr[j - 1] + dist2))))) {
                            // ins
                            cost = prev[j] + dist1;
                        } else {
                            // match
                            cost = prev[j - 1] + dist12;
                        }
                    } else {
                        cost = 0;
                    }

                    curr[j] = cost;
                    // steps[i][j] = step;
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
        }

        return Math.sqrt(curr[v2.getDimensionality() - 1]);
    }


    // utility functions, useful for cv experiments

    public static double stdv_p(Instances input) {

        double sumx = 0;
        double sumx2 = 0;
        double[] ins2array;
        for (int i = 0; i < input.numInstances(); i++) {
            ins2array = input.instance(i).toDoubleArray();
            for (int j = 0; j < ins2array.length - 1; j++) {//-1 to avoid classVal
                sumx += ins2array[j];
                sumx2 += ins2array[j] * ins2array[j];
            }
        }
        int n = input.numInstances() * (input.numAttributes() - 1);
        double mean = sumx / n;
        return Math.sqrt(sumx2 / (n) - mean * mean);

    }

    public static int[] getInclusive10(int min, int max) {
        int[] output = new int[10];

        double diff = (double) (max - min) / 9;
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

    public static double[] getInclusive10(double min, double max) {
        double[] output = new double[10];
        double diff = (double) (max - min) / 9;
        output[0] = min;
        for (int i = 1; i < 9; i++) {
            output[i] = output[i - 1] + diff;
        }
        output[9] = max;

        return output;
    }


    public static void main(String[] args) {

        ERPDistance erp = new ERPDistance(0.5, 0.5);

        double[] one = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        double[] two = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//        double[] two = {2,3,4,5,6,7,8,9,10,11};

        System.out.println(erp.distance(one, two, 0));
    }


    /************************************************************************************************
     Support for FastEE
     ************************************************************************************************/
    private final static int MAX_SEQ_LENGTH = 4000;
    private static double[] prev = new double[MAX_SEQ_LENGTH];
    private static double[] curr = new double[MAX_SEQ_LENGTH];
    private final static int[][] minDistanceToDiagonal = new int[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH];

    public static int getBandSize(double bandSize, int n) {
        return (int) Math.ceil(bandSize * n);
    }

    public static WarpingPathResults distanceExt(Instance first, Instance second, final double g, final double bandSize) {
        final int m = first.numAttributes() - 1;
        final int n = second.numAttributes() - 1;
        final int band = getBandSize(bandSize, m);
        double diff, d1, d2, d12, cost;
        int i, j, left, right, absIJ;

        Instance tmp;
        if (n < m) {
            tmp = first;
            first = second;
            second = tmp;
        }

        minDistanceToDiagonal[0][0] = 0;
        for (i = 0; i < m; i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            double[] temp = prev;
            prev = curr;
            curr = temp;

            left = i - (band + 1);
            if (left < 0) {
                left = 0;
            }
            right = i + (band + 1);
            if (right > (m - 1)) {
                right = (m - 1);
            }

            for (j = left; j <= right; j++) {
                absIJ = Math.abs(i - j);
                if (absIJ <= band) {
                    diff = first.value(i) - g;
                    d1 = (diff * diff);

                    diff = g - second.value(j);
                    d2 = (diff * diff);

                    diff = first.value(i) - second.value(j);
                    d12 = (diff * diff);

                    if ((i + j) != 0) {
                        if ((i == 0) || ((j != 0) &&
                                (((prev[j - 1] + d12) >= (curr[j - 1] + d2)) &&
                                        ((curr[j - 1] + d2) <= (prev[j] + d1))))) {
                            // del
                            cost = curr[j - 1] + d2;
                            minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i][j - 1]);
                        } else if (j == 0 || prev[j - 1] + d12 >= prev[j] + d1 && prev[j] + d1 <= curr[j - 1] + d2) {
                            // ins
                            cost = prev[j] + d1;
                            minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i - 1][j]);
                        } else {
                            // match
                            cost = prev[j - 1] + d12;
                            minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i - 1][j - 1]);
                        }
                    } else {
                        cost = 0;
                        minDistanceToDiagonal[i][j] = 0;
                    }

                    curr[j] = cost;
                    // steps[i][j] = step;
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
        }

        WarpingPathResults resExt = new WarpingPathResults();
        resExt.distance = curr[m - 1];
        resExt.distanceFromDiagonal = minDistanceToDiagonal[n - 1][m - 1];
        return resExt;
    }

}
