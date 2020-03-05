/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.classifiers.legacy.elastic_ensemble;

import experiments.data.DatasetLoading;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.CandidateNN;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.assessingNN.LazyAssessNN_WDTW;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds.LbWdtw;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import machine_learning.classifiers.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.WeightedDTW;

import java.util.ArrayList;
import java.util.Collections;
//import efficient_standalone_classifiers.Eff

/**
 * written April '16 - looks good
 *
 * @author sjx07ngu
 */
public class WDTW1NN extends Efficient1NN {

    private double g = 0;

    private double[] weightVector;
    private static final double WEIGHT_MAX = 1;
    private boolean refreshWeights = true;

    public WDTW1NN(double g) {
        this.g = g;
        this.classifierIdentifier = "WDTW_1NN";
        this.allowLoocv = false;
    }

    public WDTW1NN() {
        this.g = 0;
        this.classifierIdentifier = "WDTW_1NN";
    }

    private void initWeights(int seriesLength) {
        this.weightVector = new double[seriesLength];
        double halfLength = (double) seriesLength / 2;

        for (int i = 0; i < seriesLength; i++) {
            weightVector[i] = WEIGHT_MAX / (1 + Math.exp(-g * (i - halfLength)));
        }
        refreshWeights = false;
    }

    public final double distance(Instance first, Instance second, double cutoff) {

        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if (first.classIndex() != first.numAttributes() - 1 || second.classIndex() != second.numAttributes() - 1) {
            return new WeightedDTW(g).distance(first, second, cutoff);
        }

        int m = first.numAttributes() - 1;
        int n = second.numAttributes() - 1;

        if (this.refreshWeights) {
            this.initWeights(m);
        }


        //create empty array
        double[][] distances = new double[m][n];

        //first value
        distances[0][0] = this.weightVector[0] * (first.value(0) - second.value(0)) * (first.value(0) - second.value(0));

        //early abandon if first values is larger than cut off
        if (distances[0][0] > cutoff) {
            return Double.MAX_VALUE;
        }

        //top row
        for (int i = 1; i < n; i++) {
            distances[0][i] = distances[0][i - 1] + this.weightVector[i] * (first.value(0) - second.value(i)) * (first.value(0) - second.value(i)); //edited by Jay
        }

        //first column
        for (int i = 1; i < m; i++) {
            distances[i][0] = distances[i - 1][0] + this.weightVector[i] * (first.value(i) - second.value(0)) * (first.value(i) - second.value(0)); //edited by Jay
        }

        //warp rest
        double minDistance;
        for (int i = 1; i < m; i++) {
            boolean overflow = true;

            for (int j = 1; j < n; j++) {
                //calculate distances
                minDistance = Math.min(distances[i][j - 1], Math.min(distances[i - 1][j], distances[i - 1][j - 1]));
                distances[i][j] = minDistance + this.weightVector[Math.abs(i - j)] * (first.value(i) - second.value(j)) * (first.value(i) - second.value(j));

                if (overflow && distances[i][j] < cutoff) {
                    overflow = false; // because there's evidence that the path can continue
                }
            }

            //early abandon
            if (overflow) {
                return Double.MAX_VALUE;
            }
        }
        return distances[m - 1][n - 1];


    }


    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void runComparison() throws Exception {
        String tscProbDir = "C:/users/sjx07ngu/Dropbox/TSC Problems/";

//        String datasetName = "ItalyPowerDemand";
        String datasetName = "GunPoint";
//        String datasetName = "Beef";
//        String datasetName = "Coffee";
//        String datasetName = "SonyAiboRobotSurface1";

        double r = 0.1;
        Instances train = DatasetLoading.loadDataNullable(tscProbDir + datasetName + "/" + datasetName + "_TRAIN");
        Instances test = DatasetLoading.loadDataNullable(tscProbDir + datasetName + "/" + datasetName + "_TEST");

        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        WeightedDTW oldDtw = new WeightedDTW(r);
        knn.setDistanceFunction(oldDtw);
        knn.buildClassifier(train);

        // new version
        WDTW1NN dtwNew = new WDTW1NN(r);
        dtwNew.buildClassifier(train);

        int correctOld = 0;
        int correctNew = 0;

        long start, end, oldTime, newTime;
        double pred;

        // classification with old MSM class and kNN
        start = System.nanoTime();

        correctOld = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            pred = knn.classifyInstance(test.instance(i));
            if (pred == test.instance(i).classValue()) {
                correctOld++;
            }
        }
        end = System.nanoTime();
        oldTime = end - start;

        // classification with new MSM and own 1NN
        start = System.nanoTime();
        correctNew = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            pred = dtwNew.classifyInstance(test.instance(i));
            if (pred == test.instance(i).classValue()) {
                correctNew++;
            }
        }
        end = System.nanoTime();
        newTime = end - start;

        System.out.println("Comparison of MSM: " + datasetName);
        System.out.println("==========================================");
        System.out.println("Old acc:    " + ((double) correctOld / test.numInstances()));
        System.out.println("New acc:    " + ((double) correctNew / test.numInstances()));
        System.out.println("Old timing: " + oldTime);
        System.out.println("New timing: " + newTime);
        System.out.println("Relative Performance: " + ((double) newTime / oldTime));
    }


    public static void main(String[] args) throws Exception {
//        for(int i = 0; i < 10; i++){
//            runComparison();
//        }

        Instances train = DatasetLoading.loadDataNullable("C:/users/sjx07ngu/dropbox/tsc problems/SonyAiboRobotSurface1/SonyAiboRobotSurface1_TRAIN");

        Instance one, two;
        one = train.firstInstance();
        two = train.lastInstance();
        WeightedDTW wdtw;
        WDTW1NN wnn = new WDTW1NN();
        double g;
        for (int paramId = 0; paramId < 100; paramId++) {
            g = (double) paramId / 100;
            wdtw = new WeightedDTW(g);

            wnn.setParamsFromParamId(train, paramId);
            System.out.print(wdtw.distance(one, two) + "\t");
            System.out.println(wnn.distance(one, two, Double.MAX_VALUE));

        }


    }

    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
        this.g = (double) paramId / 100;
        refreshWeights = true;
    }

    @Override
    public String getParamInformationString() {
        return this.g + ",";
    }

    /************************************************************************************************
     Support for FastEE
     @author Chang Wei Tan, Monash University (chang.tan@monash.edu)
     ************************************************************************************************/
    protected static final int weightMax = 1;

    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex) {
        return LbWdtw.distance(candidate, weightVector[0], trainCache.getMax(queryIndex), trainCache.getMin(queryIndex));
    }

    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex, final double cutOffValue) {
        return LbWdtw.distance(candidate, weightVector[0], trainCache.getMax(queryIndex), trainCache.getMin(queryIndex), cutOffValue);
    }

    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex, final SequenceStatsCache cache) {
        return LbWdtw.distance(candidate, weightVector[0], cache.getMax(queryIndex), cache.getMin(queryIndex));
    }

    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex, final double cutOffValue, final SequenceStatsCache cache) {
        return LbWdtw.distance(candidate, weightVector[0], cache.getMax(queryIndex), cache.getMin(queryIndex), cutOffValue);
    }

    @Override
    public void initNNSTable(Instances train, SequenceStatsCache cache) {
        if (train.size() < 2) {
            System.err.println("[INIT-NNS-TABLE] Set is to small: " + train.size() + " sequence. At least 2 sequences needed.");
        }

        candidateNNS = new CandidateNN[nParams][train.size()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.size(); ++len) {
                candidateNNS[paramId][len] = new CandidateNN();
            }
        }
        classCounts = new int[nParams][train.size()][train.numClasses()];
        final boolean[] vectorCreated = new boolean[nParams];
        final double[][] weightVectors = new double[nParams][maxWindow];

        final LazyAssessNN_WDTW[] lazyAssessNNS = new LazyAssessNN_WDTW[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN_WDTW(cache);
        }
        final ArrayList<LazyAssessNN_WDTW> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Instance sCurrent = train.get(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNN_WDTW d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                if (!vectorCreated[paramId]) {
                    weightVector = initWeights(sCurrent.numAttributes() - 1, g, weightMax);
                    weightVectors[paramId] = weightVector;
                    vectorCreated[paramId] = true;
                }
                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        // --- Get the data
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNN_WDTW challenger = lazyAssessNNS[previous];
                        final LazyAssessNN_WDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, weightVectors[paramId]);

                        // --- Check the result
                        if (rrt == LazyAssessNN_WDTW.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.numClasses()];
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            }
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have the NN yet.
                    // Sort the challengers so we have the better chance to organize the good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN_WDTW challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNN_WDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, weightVectors[paramId]);

                        // --- Check the result
                        if (rrt == LazyAssessNN_WDTW.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            currPNN.set(previous, d, CandidateNN.Status.BC);
                            if (d < toBeat) {
                                classCounts[paramId][current] = new int[train.numClasses()];
                                classCounts[paramId][current][(int) challenger.getQuery().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][current][(int) challenger.getQuery().classValue()]++;
                            }
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeat(toBeat, weightVectors[paramId]);

                        // --- Check the result
                        if (rrt == LazyAssessNN_WDTW.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.numClasses()];
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            }
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    candidateNNS[paramId][current].set(currPNN.index, currPNN.distance, CandidateNN.Status.NN);
                }
            }
        }
    }

    @Override
    public void initApproxNNSTable(Instances train, SequenceStatsCache cache, int nSamples) {
        if (nSamples < 2) {
            System.err.println("[INIT-NNS-TABLE] Set is to small: " + nSamples + " sequence. At least 2 sequences needed.");
        }

        candidateNNS = new CandidateNN[nParams][nSamples];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < nSamples; ++len) {
                candidateNNS[paramId][len] = new CandidateNN();
            }
        }
        final boolean[] vectorCreated = new boolean[nParams];
        final double[][] weightVectors = new double[nParams][maxWindow];

        final LazyAssessNN_WDTW[] lazyAssessNNS = new LazyAssessNN_WDTW[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN_WDTW(cache);
        }
        final ArrayList<LazyAssessNN_WDTW> challengers = new ArrayList<>(nSamples);

        for (int current = 0; current < nSamples; ++current) {
            final Instance sCurrent = train.get(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < train.size(); ++previous) {
                if (previous == current) continue;

                final LazyAssessNN_WDTW d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                if (!vectorCreated[paramId]) {
                    weightVector = initWeights(sCurrent.numAttributes()-1, g, weightMax);
                    weightVectors[paramId] = weightVector;
                    vectorCreated[paramId] = true;
                }
                final CandidateNN currPNN = candidateNNS[paramId][current];

                Collections.sort(challengers);

                for (LazyAssessNN_WDTW challenger : challengers) {
                    // --- Get the data
                    int previous = challenger.indexQuery;
                    if (previous == current) previous = challenger.indexReference;
                    if (previous == currPNN.index) continue;

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    LazyAssessNN_WDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, weightVectors[paramId]);

                    // --- Check the result
                    if (rrt == LazyAssessNN_WDTW.RefineReturnType.New_best) {
                        double d = challenger.getDistance();
                        currPNN.set(previous, d, CandidateNN.Status.BC);
                    }

                    if (previous < nSamples) {
                        CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeat(toBeat, weightVectors[paramId]);

                        // --- Check the result
                        if (rrt == LazyAssessNN_WDTW.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
                        }
                    }
                }
            }
        }
    }

    private double[] initWeights(final int seriesLength, final double g, final double maxWeight) {
        final double[] weightVector = new double[seriesLength];
        double halfLength = (double) seriesLength / 2;

        for (int i = 0; i < seriesLength; i++) {
            weightVector[i] = maxWeight / (1 + Math.exp(-g * (i - halfLength)));
        }
        return weightVector;
    }

    public String toString() {
        return "this weight: " + this.g;
    }


}
