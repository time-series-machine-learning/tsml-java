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
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.assessingNN.LazyAssessNN_MSM;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds.LbMsm;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.MSMDistance;
import machine_learning.classifiers.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;

/**
 * checked April '16
 *
 * @author sjx07ngu
 */
//public class MSM1NN implements Classifier{
public class MSM1NN extends Efficient1NN {

    private Instances train = null;
    private double c = 0;

    protected static double[] msmParams = {
            // <editor-fold defaultstate="collapsed" desc="hidden for space">
            0.01,
            0.01375,
            0.0175,
            0.02125,
            0.025,
            0.02875,
            0.0325,
            0.03625,
            0.04,
            0.04375,
            0.0475,
            0.05125,
            0.055,
            0.05875,
            0.0625,
            0.06625,
            0.07,
            0.07375,
            0.0775,
            0.08125,
            0.085,
            0.08875,
            0.0925,
            0.09625,
            0.1,
            0.136,
            0.172,
            0.208,
            0.244,
            0.28,
            0.316,
            0.352,
            0.388,
            0.424,
            0.46,
            0.496,
            0.532,
            0.568,
            0.604,
            0.64,
            0.676,
            0.712,
            0.748,
            0.784,
            0.82,
            0.856,
            0.892,
            0.928,
            0.964,
            1,
            1.36,
            1.72,
            2.08,
            2.44,
            2.8,
            3.16,
            3.52,
            3.88,
            4.24,
            4.6,
            4.96,
            5.32,
            5.68,
            6.04,
            6.4,
            6.76,
            7.12,
            7.48,
            7.84,
            8.2,
            8.56,
            8.92,
            9.28,
            9.64,
            10,
            13.6,
            17.2,
            20.8,
            24.4,
            28,
            31.6,
            35.2,
            38.8,
            42.4,
            46,
            49.6,
            53.2,
            56.8,
            60.4,
            64,
            67.6,
            71.2,
            74.8,
            78.4,
            82,
            85.6,
            89.2,
            92.8,
            96.4,
            100// </editor-fold>
    };

    public MSM1NN() {
        this.c = 0.1;
        this.classifierIdentifier = "MSM_1NN";
    }

    public MSM1NN(double c) {
        this.c = c;
        this.classifierIdentifier = "MSM_1NN";
        this.allowLoocv = false;
    }


    public double distance(Instance first, Instance second, double cutOffValue) {

        // need to remove class index/ignore
        // simple check - if its last, ignore it. If it's not last, copy the instances, remove that attribue, and then call again

        // Not particularly efficient in the latter case, but a reasonable assumption to make here since all of the UCR/UEA problems
        // match that format. 

        int m, n;
        if (first.classIndex() == first.numAttributes() - 1 && second.classIndex() == second.numAttributes() - 1) {
            m = first.numAttributes() - 1;
            n = second.numAttributes() - 1;
        } else {
            // default case, use the original MSM class (horrible efficiency, but just in as a fail safe for edge-cases) 
            System.err.println("Warning: class designed to use problems with class index as last attribute. Defaulting to original MSM distance");
            MSMDistance msm = new MSMDistance(this.c);
            return new MSMDistance(this.c).distance(first, second);
        }

        double[][] cost = new double[m][n];

        // Initialization
        cost[0][0] = Math.abs(first.value(0) - second.value(0));
        for (int i = 1; i < m; i++) {
            cost[i][0] = cost[i - 1][0] + calcualteCost(first.value(i), first.value(i - 1), second.value(0));
        }
        for (int i = 1; i < n; i++) {
            cost[0][i] = cost[0][i - 1] + calcualteCost(second.value(i), first.value(0), second.value(i - 1));
        }

        // Main Loop
        double min;
        for (int i = 1; i < m; i++) {
            min = cutOffValue;
            for (int j = 1; j < n; j++) {
                double d1, d2, d3;
                d1 = cost[i - 1][j - 1] + Math.abs(first.value(i) - second.value(j));
                d2 = cost[i - 1][j] + calcualteCost(first.value(i), first.value(i - 1), second.value(j));
                d3 = cost[i][j - 1] + calcualteCost(second.value(j), first.value(i), second.value(j - 1));
                cost[i][j] = Math.min(d1, Math.min(d2, d3));

                if (cost[i][j] >= cutOffValue) {
                    cost[i][j] = Double.MAX_VALUE;
                }

                if (cost[i][j] < min) {
                    min = cost[i][j];
                }
            }
            if (min >= cutOffValue) {
                return Double.MAX_VALUE;
            }
        }
        // Output
        return cost[m - 1][n - 1];
    }


    public double calcualteCost(double new_point, double x, double y) {

        double dist = 0;

        if (((x <= new_point) && (new_point <= y))
                || ((y <= new_point) && (new_point <= x))) {
            dist = c;
        } else {
            dist = c + Math.min(Math.abs(new_point - x), Math.abs(new_point - y));
        }

        return dist;
    }


    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void runComparison() throws Exception {
        String tscProbDir = "C:/users/sjx07ngu/Dropbox/TSC Problems/";

//        String datasetName = "ItalyPowerDemand";
//        String datasetName = "GunPoint";
//        String datasetName = "Beef";
//        String datasetName = "Coffee";
        String datasetName = "SonyAiboRobotSurface1";

        double c = 0.1;
        Instances train = DatasetLoading.loadDataNullable(tscProbDir + datasetName + "/" + datasetName + "_TRAIN");
        Instances test = DatasetLoading.loadDataNullable(tscProbDir + datasetName + "/" + datasetName + "_TEST");

        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        MSMDistance msmOld = new MSMDistance(c);
        knn.setDistanceFunction(msmOld);
        knn.buildClassifier(train);

        // new version
        MSM1NN msmNew = new MSM1NN(c);
        msmNew.buildClassifier(train);

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
            pred = msmNew.classifyInstance(test.instance(i));
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
//        
        for (int i = 0; i < 10; i++) {
            runComparison();
        }
    }

    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
        this.c = msmParams[paramId];
    }

    @Override
    public String getParamInformationString() {
        return this.c + "";
    }

    /************************************************************************************************
     Support for FastEE
     @author Chang Wei Tan, Monash University (chang.tan@monash.edu)
     ************************************************************************************************/
    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex){
        return LbMsm.distance(query, candidate, c, trainCache.getMax(queryIndex), trainCache.getMin(queryIndex));
    }

    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex, final double cutOffValue){
        return LbMsm.distance(query, candidate, c, trainCache.getMax(queryIndex), trainCache.getMin(queryIndex), cutOffValue);
    }

    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex, final SequenceStatsCache cache){
        return LbMsm.distance(query, candidate, c, cache.getMax(queryIndex), cache.getMin(queryIndex));
    }

    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex, final double cutOffValue,  final SequenceStatsCache cache){
        return LbMsm.distance(query, candidate, c, cache.getMax(queryIndex), cache.getMin(queryIndex), cutOffValue);
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

        final LazyAssessNN_MSM[] lazyAssessNNS = new LazyAssessNN_MSM[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN_MSM(cache);
        }
        final ArrayList<LazyAssessNN_MSM> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Instance sCurrent = train.get(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNN_MSM d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = 0; paramId < nParams; ++paramId) {
                setParamsFromParamId(train, paramId);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have NN for sure, but we still have to check if current is new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNN_MSM challenger = lazyAssessNNS[previous];
                        final LazyAssessNN_MSM.RefineReturnType rrt = challenger.tryToBeat(toBeat, c);

                        // --- Check the result
                        if (rrt == LazyAssessNN_MSM.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
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
                    // We don't have NN yet.
                    // Sort the challengers so we have better chance to organize good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN_MSM challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNN_MSM.RefineReturnType rrt = challenger.tryToBeat(toBeat, c);

                        // --- Check the result
                        if (rrt == LazyAssessNN_MSM.RefineReturnType.New_best) {
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
                        rrt = challenger.tryToBeat(toBeat, c);

                        // --- Check the result
                        if (rrt == LazyAssessNN_MSM.RefineReturnType.New_best) {
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
                    if (paramId > 0) {
                        final double currD = currPNN.distance;
                        final double prevD = candidateNNS[paramId - 1][current].distance;
                        final int index = currPNN.index;
                        if (currD == prevD) {
                            for (int j = paramId; j < nParams; j++) {
                                candidateNNS[j][current].set(index, currD, CandidateNN.Status.NN);
                                classCounts[j][current] = classCounts[paramId][current].clone();
                            }
                        }
                    }
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

        final LazyAssessNN_MSM[] lazyAssessNNS = new LazyAssessNN_MSM[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN_MSM(cache);
        }
        final ArrayList<LazyAssessNN_MSM> challengers = new ArrayList<>(nSamples);

        for (int current = 0; current < nSamples; ++current) {
            final Instance sCurrent = train.get(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < train.size(); ++previous) {
                if (previous==current) continue;

                final LazyAssessNN_MSM d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = 0; paramId < nParams; ++paramId) {
                setParamsFromParamId(train, paramId);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                Collections.sort(challengers);
                boolean newNN = false;
                for (LazyAssessNN_MSM challenger : challengers) {
                    // --- Get the data
                    int previous = challenger.indexQuery;
                    if (previous == current) previous = challenger.indexReference;
                    if (previous == currPNN.index) continue;

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    LazyAssessNN_MSM.RefineReturnType rrt = challenger.tryToBeat(toBeat, c);

                    // --- Check the result
                    if (rrt == LazyAssessNN_MSM.RefineReturnType.New_best) {
                        double d = challenger.getDistance();
                        currPNN.set(previous, d, CandidateNN.Status.BC);
                        newNN = true;
                    }

                    if (previous < nSamples) {
                        CandidateNN prevNN = candidateNNS[paramId][previous];
                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeat(toBeat, c);

                        // --- Check the result
                        if (rrt == LazyAssessNN_MSM.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
                        }
                    }
                }

                if (newNN && paramId > 0) {
                    double currD = currPNN.distance;
                    double prevD = candidateNNS[paramId - 1][current].distance;
                    int index = currPNN.index;
                    if (currD == prevD) {
                        for (int j = paramId; j < nParams; j++) {
                            candidateNNS[j][current].set(index, currD, CandidateNN.Status.NN);
                        }
                    }
                }
            }
        }
    }
}
