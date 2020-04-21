/*******************************************************************************
 * Copyright (C) 2017 Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 *
 * This file is part of FastWWSearch.
 *
 * FastWWSearch is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * FastWWSearch is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with FastWWSearch.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils;

import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds.LbErp;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds.LbKeogh;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds.LbLcss;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * <p>
 * Cache for storing the information on the time series dataset
 *
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 */
public class SequenceStatsCache {
    protected ArrayList<double[]> LEs, UEs;
    protected double[] mins, maxs;
    protected int[] indexMaxs, indexMins;
    protected boolean[] isMinFirst, isMinLast, isMaxFirst, isMaxLast;
    protected double[] lastWindowComputed;
    protected double[] lastERPWindowComputed;
    protected double[] lastLCSSWindowComputed;
    protected int currentWindow;
    protected Instances train;
    protected IndexedDouble[][] indicesSortedByAbsoluteValue;
    protected double[][] lbDistances;

    public SequenceStatsCache(final Instances train, final int startingWindow) {
        this.train = train;
        int nSequences = train.size();
        int length = train.numAttributes() - 1;
        this.LEs = new ArrayList<>(nSequences);
        this.UEs = new ArrayList<>(nSequences);
        this.lastWindowComputed = new double[nSequences];
        this.lastERPWindowComputed = new double[nSequences];
        this.lastLCSSWindowComputed = new double[nSequences];
        Arrays.fill(this.lastWindowComputed, -1);
        Arrays.fill(this.lastERPWindowComputed, -1);
        Arrays.fill(this.lastLCSSWindowComputed, -1);
        this.currentWindow = startingWindow;
        this.mins = new double[nSequences];
        this.maxs = new double[nSequences];
        this.indexMins = new int[nSequences];
        this.indexMaxs = new int[nSequences];
        this.isMinFirst = new boolean[nSequences];
        this.isMinLast = new boolean[nSequences];
        this.isMaxFirst = new boolean[nSequences];
        this.isMaxLast = new boolean[nSequences];
        this.indicesSortedByAbsoluteValue = new IndexedDouble[nSequences][length];
        for (int i = 0; i < train.size(); i++) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            int indexMin = -1, indexMax = -1;
            for (int j = 0; j < train.numAttributes() - 1; j++) {
                double val = train.get(i).value(j);
                if (val > max) {
                    max = val;
                    indexMax = j;
                }
                if (val < min) {
                    min = val;
                    indexMin = j;
                }
                indicesSortedByAbsoluteValue[i][j] = new IndexedDouble(j, Math.abs(val));
            }
            indexMaxs[i] = indexMax;
            indexMins[i] = indexMin;
            mins[i] = min;
            maxs[i] = max;
            isMinFirst[i] = (indexMin == 0);
            isMinLast[i] = (indexMin == (train.numAttributes() - 2));
            isMaxFirst[i] = (indexMax == 0);
            isMaxLast[i] = (indexMax == (train.numAttributes() - 2));
            Arrays.sort(indicesSortedByAbsoluteValue[i], (v1, v2) -> -Double.compare(v1.value, v2.value));
            this.LEs.add(new double[length]);
            this.UEs.add(new double[length]);
        }
    }

    public double[] getLE(final int i, final int w) {
        if (lastWindowComputed[i] != w) {
            LEs.add(new double[train.get(i).numAttributes() - 1]);
            UEs.add(new double[train.get(i).numAttributes() - 1]);
            computeLEandUE(i, w);
        }
        return LEs.get(i);
    }

    public double[] getUE(final int i, final int w) {
        if (lastWindowComputed[i] != w) {
            LEs.add(new double[train.get(i).numAttributes() - 1]);
            UEs.add(new double[train.get(i).numAttributes() - 1]);
            computeLEandUE(i, w);
        }
        return UEs.get(i);
    }

    public void computeLEandUE(final int i, final int r) {
        LbKeogh.fillUL(train.get(i), r, UEs.get(i), LEs.get(i));
        this.lastWindowComputed[i] = r;
    }

    public double[] getLE(final int i, final double g, final double bandSize) {
        if (lastERPWindowComputed[i] != bandSize) {
            LEs.add(new double[train.get(i).numAttributes()-1]);
            UEs.add(new double[train.get(i).numAttributes()-1]);
            computeLEandUE(i, g, bandSize);
        }
        return LEs.get(i);
    }

    public double[] getUE(final int i, final double g, final double bandSize) {
        if (lastERPWindowComputed[i] != bandSize) {
            LEs.add(new double[train.get(i).numAttributes()-1]);
            UEs.add(new double[train.get(i).numAttributes()-1]);
            computeLEandUE(i, g, bandSize);
        }
        return UEs.get(i);
    }

    public void computeLEandUE(final int i, final double g, final double bandSize) {
        LbErp.fillUL(train.get(i), g, bandSize, UEs.get(i), LEs.get(i));
        this.lastERPWindowComputed[i] = bandSize;
    }

    public double[] getLE(final int i, final int delta, final double epsilon) {
        if (lastLCSSWindowComputed[i] != delta) {
            LEs.add(new double[train.get(i).numAttributes()-1]);
            UEs.add(new double[train.get(i).numAttributes()-1]);
            computeLEandUE(i, delta, epsilon);
        }
        return LEs.get(i);
    }

    public double[] getUE(final int i, final int delta, final double epsilon) {
        if (lastLCSSWindowComputed[i] != delta) {
            LEs.add(new double[train.get(i).numAttributes()-1]);
            UEs.add(new double[train.get(i).numAttributes()-1]);
            computeLEandUE(i, delta, epsilon);
        }
        return UEs.get(i);
    }

    public void computeLEandUE(final int i, final int delta, final double epsilon) {
        LbLcss.fillUL(train.get(i), epsilon, delta, UEs.get(i), LEs.get(i));
        this.lastLCSSWindowComputed[i] = delta;
    }

    public boolean isMinFirst(int i) {
        return isMinFirst[i];
    }

    public boolean isMaxFirst(int i) {
        return isMaxFirst[i];
    }

    public boolean isMinLast(int i) {
        return isMinLast[i];
    }

    public boolean isMaxLast(int i) {
        return isMaxLast[i];
    }

    public double getMin(int i) {
        return mins[i];
    }

    public double getMax(int i) {
        return maxs[i];
    }

    public int getIMax(int i) {
        return indexMaxs[i];
    }

    public int getIMin(int i) {
        return indexMins[i];
    }

    public int getIndexNthHighestVal(int i, int n) {
        return indicesSortedByAbsoluteValue[i][n].index;
    }

    public void initLbDistances() {
        lbDistances = new double[train.size()][train.size()];
    }

    public void setLbDistances(double lb, int qIndex, int cIndex) {
        lbDistances[qIndex][cIndex] = lb;
    }

    public double getLbDistances(int qIndex, int cIndex) {
        return lbDistances[qIndex][cIndex];
    }

    public boolean lbDistanceExist(int qIndex, int cIndex) {
        return lbDistances[qIndex][cIndex] != 0;
    }
}
