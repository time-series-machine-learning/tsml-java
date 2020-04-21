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
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.assessingNN.LazyAssessNN_ERP;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds.LbErp;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.ERPDistance;
import machine_learning.classifiers.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
//import efficient_standalone_classifiers.Eff

/**
 * adjusted April '16
 * note: not using DTW class in here (redoing the method) as even though the DTW class is already about as efficient, it still
 * involves some array copying. Here we can opperate straight on the Instance values instead
 *
 * @author sjx07ngu
 */
public class ERP1NN extends Efficient1NN {

    private double g;
    private double bandSize;

    private double[] gValues;
    private double[] windowSizes;
    private boolean gAndWindowsRefreshed = false;

    public ERP1NN(double g, double bandSize) {
        this.g = g;
        this.bandSize = bandSize;
        this.gAndWindowsRefreshed = false;
        this.classifierIdentifier = "ERP_1NN";
        this.allowLoocv = false;
    }


    public ERP1NN() {
        // note: default params probably won't suit the majority of problems. Should set through cv or prior knowledge
        this.g = 0.5;
        this.bandSize = 5;
        this.gAndWindowsRefreshed = false;
        this.classifierIdentifier = "ERP_1NN";
    }

    @Override
    public void buildClassifier(Instances train) throws Exception {
        super.buildClassifier(train);
        this.gAndWindowsRefreshed = false;

    }


    public final double distance(Instance first, Instance second, double cutoff) {

        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if (first.classIndex() != first.numAttributes() - 1 || second.classIndex() != second.numAttributes() - 1) {
            return new ERPDistance(this.g, this.bandSize).distance(first, second, cutoff);
        }

        int m = first.numAttributes() - 1;
        int n = second.numAttributes() - 1;


        // Current and previous columns of the matrix
        double[] curr = new double[m];
        double[] prev = new double[m];

        // size of edit distance band
        // bandsize is the maximum allowed distance to the diagonal
//        int band = (int) Math.ceil(v2.getDimensionality() * bandSize);
        int band = (int) Math.ceil(m * bandSize);

        // g parameter for local usage
        double gValue = g;

        for (int i = 0; i < m; i++) {
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
            if (r > (m - 1)) {
                r = (m - 1);
            }

            for (int j = l; j <= r; j++) {
                if (Math.abs(i - j) <= band) {
                    // compute squared distance of feature vectors
                    double val1 = first.value(i);
                    double val2 = gValue;
                    double diff = (val1 - val2);
                    final double d1 = Math.sqrt(diff * diff);

                    val1 = gValue;
                    val2 = second.value(j);
                    diff = (val1 - val2);
                    final double d2 = Math.sqrt(diff * diff);

                    val1 = first.value(i);
                    val2 = second.value(j);
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

        return Math.sqrt(curr[m - 1]);
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

        double r = 0.1;
        Instances train = DatasetLoading.loadDataNullable(tscProbDir + datasetName + "/" + datasetName + "_TRAIN");
        Instances test = DatasetLoading.loadDataNullable(tscProbDir + datasetName + "/" + datasetName + "_TEST");

        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        ERPDistance oldDtw = new ERPDistance(0.1, 0.1);
//        oldDtw.setR(r);
        knn.setDistanceFunction(oldDtw);
        knn.buildClassifier(train);

        // new version
        ERP1NN dtwNew = new ERP1NN(0.1, 0.1);
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
        for (int i = 0; i < 10; i++) {
            runComparison();
        }
    }

    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
        if (!this.gAndWindowsRefreshed) {
            double stdv = ERPDistance.stdv_p(train);
            windowSizes = ERPDistance.getInclusive10(0, 0.25);
            gValues = ERPDistance.getInclusive10(0.2 * stdv, stdv);
            this.gAndWindowsRefreshed = true;
        }
        this.g = gValues[paramId / 10];
        this.bandSize = windowSizes[paramId % 10];
    }

    @Override
    public String getParamInformationString() {
        return this.g + "," + this.bandSize;
    }


    /************************************************************************************************
     Support for FastEE
     @author Chang Wei Tan, Monash University (chang.tan@monash.edu)
     ************************************************************************************************/
    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex) {
        return LbErp.distance(candidate, trainCache.getUE(queryIndex, g, bandSize), trainCache.getLE(queryIndex, g, bandSize));
    }

    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex, double cutOffValue) {
        return LbErp.distance(candidate, trainCache.getUE(queryIndex, g, bandSize), trainCache.getLE(queryIndex, g, bandSize), cutOffValue);
    }

    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex, SequenceStatsCache cache) {
        return LbErp.distance(candidate, cache.getUE(queryIndex, g, bandSize), cache.getLE(queryIndex, g, bandSize));
    }

    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex, double cutOffValue, SequenceStatsCache cache) {
        return LbErp.distance(candidate, cache.getUE(queryIndex, g, bandSize), cache.getLE(queryIndex, g, bandSize), cutOffValue);
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

        final LazyAssessNN_ERP[] lazyAssessNNS = new LazyAssessNN_ERP[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN_ERP(cache);
        }
        final ArrayList<LazyAssessNN_ERP> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Instance sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNN_ERP d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        // --- Get the data
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNN_ERP challenger = lazyAssessNNS[previous];
                        final LazyAssessNN_ERP.RefineReturnType rrt = challenger.tryToBeat(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN_ERP.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(ERPDistance.getBandSize(bandSize, train.numAttributes() - 1));
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
                    // We don't have the NN yet.
                    // Sort the challengers so we have the better chance to organize the good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN_ERP challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNN_ERP.RefineReturnType rrt = challenger.tryToBeat(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN_ERP.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(ERPDistance.getBandSize(bandSize, train.numAttributes() - 1));
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
                        rrt = challenger.tryToBeat(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN_ERP.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(ERPDistance.getBandSize(bandSize, train.numAttributes() - 1));
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
                    final double prevG = g;
                    int w = ERPDistance.getBandSize(bandSize, train.numAttributes() - 1);
                    int tmp = paramId;
                    while (tmp > 0 && paramId % 10 > 0 && prevG == g && w >= r) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();
                        tmp--;
                        this.setParamsFromParamId(train, tmp);
                        w = ERPDistance.getBandSize(bandSize, train.numAttributes() - 1);
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

        final LazyAssessNN_ERP[] lazyAssessNNS = new LazyAssessNN_ERP[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN_ERP(cache);
        }
        final ArrayList<LazyAssessNN_ERP> challengers = new ArrayList<>(nSamples);

        for (int current = 0; current < nSamples; ++current) {
            final Instance sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < train.size(); ++previous) {
                if (previous == current) continue;

                final LazyAssessNN_ERP d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                Collections.sort(challengers);
                boolean newNN = false;

                for (LazyAssessNN_ERP challenger : challengers) {
                    // --- Get the data
                    int previous = challenger.indexQuery;
                    if (previous == current) previous = challenger.indexReference;
                    if (previous == currPNN.index) continue;

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    LazyAssessNN_ERP.RefineReturnType rrt = challenger.tryToBeat(toBeat, g, bandSize);

                    // --- Check the result
                    if (rrt == LazyAssessNN_ERP.RefineReturnType.New_best) {
                        int r = challenger.getMinWindowValidityForFullDistance();
                        double d = challenger.getDistance(ERPDistance.getBandSize(bandSize, train.numAttributes() - 1));
                        currPNN.set(previous, r, d, CandidateNN.Status.BC);
                        newNN = true;
                    }

                    if (previous < nSamples) {
                        CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeat(toBeat, g, bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNN_ERP.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(ERPDistance.getBandSize(bandSize, train.numAttributes() - 1));
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
                        }
                    }
                }

                if (newNN) {
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int w = ERPDistance.getBandSize(bandSize, train.numAttributes() - 1);
                    int tmp = paramId;
                    double prevG = g;
                    while (tmp >= 0 && prevG == g && w >= r) {
                        prevG = g;
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        tmp--;
                        if (tmp >= 0) {
                            this.setParamsFromParamId(train, tmp);
                            w = ERPDistance.getBandSize(bandSize, train.numAttributes() - 1);
                        }
                    }
                }
            }
        }
    }


}
