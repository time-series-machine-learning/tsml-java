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
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.assessingNN.LazyAssessNN_LCSS;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds.LbLcss;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.LCSSDistance;
import machine_learning.classifiers.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;

/**
 * checked April l16
 *
 * @author sjx07ngu
 */
public class LCSS1NN extends Efficient1NN {

    private int delta;
    private double epsilon;

    boolean epsilonsAndDeltasRefreshed;
    double[] epsilons;
    int[] deltas;

    public LCSS1NN(int delta, double epsilon) {
        this.delta = delta;
        this.epsilon = epsilon;
        epsilonsAndDeltasRefreshed = false;
        this.classifierIdentifier = "LCSS_1NN";
        this.allowLoocv = false;
    }

    public LCSS1NN() {
        // note: these default params may be garbage for most datasets, should set them through CV
        this.delta = 3;
        this.epsilon = 1;
        epsilonsAndDeltasRefreshed = false;
        this.classifierIdentifier = "LCSS_1NN";
    }

    @Override
    public void buildClassifier(Instances train) throws Exception {
        super.buildClassifier(train);

        // used for setting params with the paramId method
        epsilonsAndDeltasRefreshed = false;
    }


    public double distance(Instance first, Instance second) {

        // need to remove class index/ignore
        // simple check - if its last, ignore it. If it's not last, copy the instances, remove that attribue, and then call again 
        //  edit: can't do a simple copy with Instance objs by the looks of things. Fail-safe: fall back to the original measure

        int m, n;
        if (first.classIndex() == first.numAttributes() - 1 && second.classIndex() == second.numAttributes() - 1) {
            m = first.numAttributes() - 1;
            n = second.numAttributes() - 1;
        } else {
            // default case, use the original MSM class (horrible efficiency, but just in as a fail safe for edge-cases) 
            System.err.println("Warning: class designed to use problems with class index as last attribute. Defaulting to original MSM distance");
            return new LCSSDistance(this.delta, this.epsilon).distance(first, second);
        }

        int[][] lcss = new int[m + 1][n + 1];

        for (int i = 0; i < m; i++) {
            for (int j = i - delta; j <= i + delta; j++) {
                if (j < 0) {
                    j = -1;
                } else if (j >= n) {
                    j = i + delta;
                } else if (second.value(j) + this.epsilon >= first.value(i) && second.value(j) - epsilon <= first.value(i)) {
                    lcss[i + 1][j + 1] = lcss[i][j] + 1;
                } else if (lcss[i][j + 1] > lcss[i + 1][j]) {
                    lcss[i + 1][j + 1] = lcss[i][j + 1];
                } else {
                    lcss[i + 1][j + 1] = lcss[i + 1][j];
                }

                // could maybe do an early abandon here? Not sure, investigate further 
            }
        }

        int max = -1;
        for (int i = 1; i < lcss[lcss.length - 1].length; i++) {
            if (lcss[lcss.length - 1][i] > max) {
                max = lcss[lcss.length - 1][i];
            }
        }
        return 1 - ((double) max / m);

    }


    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void main(String[] args) throws Exception {
        for (int i = 0; i < 10; i++) {
            runComparison();
        }
    }

    public static void runComparison() throws Exception {
        String tscProbDir = "C:/users/sjx07ngu/Dropbox/TSC Problems/";

//        String datasetName = "ItalyPowerDemand";
        String datasetName = "GunPoint";
//        String datasetName = "Beef";
//        String datasetName = "Coffee";
//        String datasetName = "SonyAiboRobotSurface1";


        Instances train = DatasetLoading.loadDataNullable(tscProbDir + datasetName + "/" + datasetName + "_TRAIN");
        Instances test = DatasetLoading.loadDataNullable(tscProbDir + datasetName + "/" + datasetName + "_TEST");

        int delta = 10;
        double epsilon = 0.5;


        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        LCSSDistance lcssOld = new LCSSDistance(delta, epsilon);
        knn.setDistanceFunction(lcssOld);
        knn.buildClassifier(train);

        // new version
        LCSS1NN lcssNew = new LCSS1NN(delta, epsilon);
        lcssNew.buildClassifier(train);

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

        // classification with new MSM and in-build 1NN
        start = System.nanoTime();
        correctNew = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            pred = lcssNew.classifyInstance(test.instance(i));
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

    @Override
    public double distance(Instance first, Instance second, double cutOffValue) {
//        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        return this.distance(first, second);
    }


    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
        // more efficient to only calculate these when the training data has been changed, so could call in build classifier
        // however, these values are only needed in this method, so calculate here. 
        // If the training data hasn't changed (i.e. no calls to buildClassifier, then they don't need recalculated 
        if (!epsilonsAndDeltasRefreshed) {
            double stdTrain = LCSSDistance.stdv_p(train);
            double stdFloor = stdTrain * 0.2;
            epsilons = LCSSDistance.getInclusive10(stdFloor, stdTrain);
            deltas = LCSSDistance.getInclusive10(0, (train.numAttributes() - 1) / 4);
            epsilonsAndDeltasRefreshed = true;
        }
        this.delta = deltas[paramId / 10];
        this.epsilon = epsilons[paramId % 10];
    }

    @Override
    public String getParamInformationString() {
        return this.delta + "," + this.epsilon;
    }

    /************************************************************************************************
     Support for FastEE
     @author Chang Wei Tan, Monash University (chang.tan@monash.edu)
     ************************************************************************************************/
    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex) {
        return LbLcss.distance(candidate, trainCache.getUE(queryIndex, delta, epsilon), trainCache.getLE(queryIndex, delta, epsilon));
    }

    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex, double cutOffValue) {
        return LbLcss.distance(candidate, trainCache.getUE(queryIndex, delta, epsilon), trainCache.getLE(queryIndex, delta, epsilon), cutOffValue);
    }

    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex, SequenceStatsCache cache) {
        return LbLcss.distance(candidate, cache.getUE(queryIndex, delta, epsilon), cache.getLE(queryIndex, delta, epsilon));
    }

    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex, double cutOffValue, SequenceStatsCache cache) {
        return LbLcss.distance(candidate, cache.getUE(queryIndex, delta, epsilon), cache.getLE(queryIndex, delta, epsilon), cutOffValue);
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

        final LazyAssessNN_LCSS[] lazyAssessNNS = new LazyAssessNN_LCSS[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN_LCSS(cache);
        }
        final ArrayList<LazyAssessNN_LCSS> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Instance sCurrent = train.get(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNN_LCSS d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have  NN for sure, but we still have to check if current is  new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        double toBeat = prevNN.distance;
                        final LazyAssessNN_LCSS challenger = lazyAssessNNS[previous];
                        final LazyAssessNN_LCSS.RefineReturnType rrt = challenger.tryToBeat(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN_LCSS.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(delta);
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
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
                    // We don't have  NN yet.
                    // Sort the challengers so we have  better chance to organize  good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN_LCSS challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate of reference:
                        double toBeat = currPNN.distance;
                        LazyAssessNN_LCSS.RefineReturnType rrt = challenger.tryToBeat(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN_LCSS.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(delta);
                            currPNN.set(previous, r, d, CandidateNN.Status.BC);
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
                        rrt = challenger.tryToBeat(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN_LCSS.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(delta);
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
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
                    final int r = currPNN.r;
                    final double d = currPNN.distance;
                    final int index = currPNN.index;
                    final double prevEpsilon = epsilon;
                    int tmp = paramId;
                    while (tmp > 0 && paramId % 10 > 0 && prevEpsilon == epsilon && delta >= r) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();
                        tmp--;
                        this.setParamsFromParamId(train, tmp);
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
        classCounts = new int[nParams][nSamples][train.numClasses()];

        final LazyAssessNN_LCSS[] lazyAssessNNS = new LazyAssessNN_LCSS[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN_LCSS(cache);
        }
        final ArrayList<LazyAssessNN_LCSS> challengers = new ArrayList<>(nSamples);

        for (int current = 0; current < nSamples; ++current) {
            final Instance sCurrent = train.get(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < train.size(); ++previous) {
                if (previous == current) continue;

                final LazyAssessNN_LCSS d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                Collections.sort(challengers);
                boolean newNN = false;
                for (LazyAssessNN_LCSS challenger : challengers) {
                    // --- Get the data
                    int previous = challenger.indexQuery;
                    if (previous == current) previous = challenger.indexReference;
                    if (previous == currPNN.index) continue;

                    // --- First we want to beat the current best candidate of reference:
                    double toBeat = currPNN.distance;
                    LazyAssessNN_LCSS.RefineReturnType rrt = challenger.tryToBeat(toBeat, delta, epsilon);

                    // --- Check the result
                    if (rrt == LazyAssessNN_LCSS.RefineReturnType.New_best) {
                        int r = challenger.getMinWindowValidityForFullDistance();
                        double d = challenger.getDistance(delta);
                        currPNN.set(previous, r, d, CandidateNN.Status.BC);
                        if (d < toBeat) {
                            classCounts[paramId][current] = new int[train.numClasses()];
                            classCounts[paramId][current][(int) challenger.getQuery().classValue()]++;
                        } else if (d == toBeat) {
                            classCounts[paramId][current][(int) challenger.getQuery().classValue()]++;
                        }
                        newNN = true;
                    }

                    if (previous < nSamples) {
                        CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeat(toBeat, delta, epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNN_LCSS.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(delta);
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.numClasses()];
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][(int) challenger.getReference().classValue()]++;
                            }
                        }
                    }
                }

                if (newNN) {
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int tmp = paramId;
                    candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                    double prevEpsilon = epsilon;
                    while (tmp > 0 && paramId % 10 > 0 && prevEpsilon == epsilon && delta >= r) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();

                        tmp--;
                        this.setParamsFromParamId(train, tmp);
                    }
                }
            }
        }
    }


}
