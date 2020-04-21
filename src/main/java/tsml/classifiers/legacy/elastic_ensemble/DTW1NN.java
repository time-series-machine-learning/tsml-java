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
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.assessingNN.LazyAssessNN_DTW;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds.LbKeogh;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.DTW;
import machine_learning.classifiers.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;

/**
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class DTW1NN extends Efficient1NN {


    private double r = 1;

    private int window;

    /**
     * Constructor with specified window size (between 0 and 1). When a window
     * size is specified, cross-validation methods will become inactive for this
     * object.
     * <p>
     * Note: if window = 1, classifierIdentifier will be DTW_R1_1NN; other
     * window sizes will results in cId of DTW_Rn_1NN instead. This information
     * is used for any file writing
     *
     * @param r
     */
    public DTW1NN(double r) {
        this.allowLoocv = false;
        this.r = r;
        if (r != 1) {
            this.classifierIdentifier = "DTW_Rn_1NN";
        } else {
            this.classifierIdentifier = "DTW_R1_1NN";
        }
    }

    /**
     * A default constructor. Sets the window to 1 (100%), but allows for the
     * option of cross-validation if the relevant method is called.
     * <p>
     * classifierIdentifier is initially set to DTW_R1_1NN, but will
     * update automatically to DTW_Rn_1NN if loocv is called
     */
    public DTW1NN() {
        this.r = 1;
        this.classifierIdentifier = "DTW_R1_1NN";
    }

    public void setWindow(double w) {
        r = w;
    }

    public void turnOffCV() {
        this.allowLoocv = false;
    }

    public void turnOnCV() {
        this.allowLoocv = true;
    }

    @Override
    public double[] loocv(Instances train) throws Exception {
        if (this.allowLoocv == true && this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
        }
        return super.loocv(train);
    }

    @Override
    public double[] loocv(Instances[] train) throws Exception {
        if (this.allowLoocv == true && this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
        }
        return super.loocv(train);
    }


    final public int getWindowSize(int n) {
        window = (int) (r * n);   //Rounded down.
        //No Warp, windowSize=1
        if (window < 1) window = 1;
            //Full Warp : windowSize=n, otherwise scale between
        else if (window < n)
            window++;
        return window;
    }


//    public double classifyInstance(Instance instance) throws Exception{
//        if(isDerivative){
//            Instances temp = new Instances(instance.dataset(),1);
//            temp.add(instance);
//            temp = new Derivative().process(temp);
//            return classifyInstance(temp.instance(0));
//        }
//        return super.classifyInstance(instance);
//    }

    public final double distance(Instance first, Instance second, double cutoff) {

        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if (first.classIndex() != first.numAttributes() - 1 || second.classIndex() != second.numAttributes() - 1) {
            DTW temp = new DTW();
            temp.setR(r);
            return temp.distance(first, second, cutoff);
        }

        double minDist;
        boolean tooBig;

        int n = first.numAttributes() - 1;
        int m = second.numAttributes() - 1;
        /*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp 
         generalised for variable window size
         * */
        int windowSize = getWindowSize(n);
//Extra memory than required, could limit to windowsize,
//        but avoids having to recreate during CV 
//for varying window sizes        
        double[][] matrixD = new double[n][m];
        
        /*
         //Set boundary elements to max. 
         */
        int start, end;
        for (int i = 0; i < n; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = i + windowSize + 1 < m ? i + windowSize + 1 : m;
            for (int j = start; j < end; j++) {
                matrixD[i][j] = Double.MAX_VALUE;
            }
        }
        matrixD[0][0] = (first.value(0) - second.value(0)) * (first.value(0) - second.value(0));
//a is the longer series. 
//Base cases for warping 0 to all with max interval	r	
//Warp first.value(0] onto all second.value(1]...second.value(r+1]
        for (int j = 1; j < windowSize && j < m; j++) {
            matrixD[0][j] = matrixD[0][j - 1] + (first.value(0) - second.value(j)) * (first.value(0) - second.value(j));
        }

//	Warp second.value(0] onto all first.value(1]...first.value(r+1]
        for (int i = 1; i < windowSize && i < n; i++) {
            matrixD[i][0] = matrixD[i - 1][0] + (first.value(i) - second.value(0)) * (first.value(i) - second.value(0));
        }
//Warp the rest,
        for (int i = 1; i < n; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = i + windowSize < m ? i + windowSize : m;
            for (int j = start; j < end; j++) {
                minDist = matrixD[i][j - 1];
                if (matrixD[i - 1][j] < minDist) {
                    minDist = matrixD[i - 1][j];
                }
                if (matrixD[i - 1][j - 1] < minDist) {
                    minDist = matrixD[i - 1][j - 1];
                }
                matrixD[i][j] = minDist + (first.value(i) - second.value(j)) * (first.value(i) - second.value(j));
                if (tooBig && matrixD[i][j] < cutoff) {
                    tooBig = false;
                }
            }
            //Early abandon
            if (tooBig) {
                return Double.MAX_VALUE;
            }
        }
//Find the minimum distance at the end points, within the warping window. 
        return matrixD[n - 1][m - 1];
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] res = new double[instance.numClasses()];
        int r = (int) classifyInstance(instance);
        res[r] = 1;
        return res;
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
        DTW oldDtw = new DTW();
        oldDtw.setR(r);
        knn.setDistanceFunction(oldDtw);
        knn.buildClassifier(train);

        // new version
        DTW1NN dtwNew = new DTW1NN(r);
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
        if (this.allowLoocv) {
            if (this.classifierIdentifier.contains("R1")) {
                this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
            }
            this.r = (double) paramId / 100;
        } else {
            throw new RuntimeException("Warning: trying to set parameters of a fixed window DTW");
        }
    }

    @Override
    public String getParamInformationString() {
        return this.r + "";
    }


    /************************************************************************************************
     Support for FastEE
     @author Chang Wei Tan, Monash University (chang.tan@monash.edu)
     ************************************************************************************************/
    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex) {
        return LbKeogh.distance(candidate, trainCache.getUE(queryIndex, window), trainCache.getLE(queryIndex, window));
    }

    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex, double cutOffValue) {
        return LbKeogh.distance(candidate, trainCache.getUE(queryIndex, window), trainCache.getLE(queryIndex, window), cutOffValue);
    }

    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex, SequenceStatsCache cache) {
        return LbKeogh.distance(candidate, cache.getUE(queryIndex, window), cache.getLE(queryIndex, window));
    }

    @Override
    public double lowerBound(Instance query, Instance candidate, int queryIndex, int candidateIndex, double cutOffValue, SequenceStatsCache cache) {
        return LbKeogh.distance(candidate, cache.getUE(queryIndex, window), cache.getLE(queryIndex, window), cutOffValue);
    }


    @Override
    public void initNNSTable(Instances train, SequenceStatsCache cache) {
        if (train.size() < 2) {
            System.err.println("[INIT-NNS-TABLE] Set is too small: " + train.size() + " sequence. At least 2 sequences needed.");
        }

        candidateNNS = new CandidateNN[nParams][train.size()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.size(); ++len) {
                candidateNNS[paramId][len] = new CandidateNN();
            }
        }
        classCounts = new int[nParams][train.size()][train.numClasses()];

        final LazyAssessNN_DTW[] lazyAssessNNS = new LazyAssessNN_DTW[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN_DTW(cache);
        }
        final ArrayList<LazyAssessNN_DTW> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Instance sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNN_DTW d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                final int win = getWindowSize(maxWindow);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNN_DTW challenger = lazyAssessNNS[previous];
                        final LazyAssessNN_DTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN_DTW.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(win);
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

                    for (LazyAssessNN_DTW challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNN_DTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN_DTW.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(win);
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
                        rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN_DTW.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(win);
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
                    final int winEnd = getParamIdFromWindow(r, train.numAttributes() - 1);
                    for (int tmp = paramId; tmp >= winEnd; --tmp) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();
                    }
                }
            }
        }
    }

    @Override
    public void initApproxNNSTable(Instances train, SequenceStatsCache cache, int nSamples) {
        if (nSamples < 2) {
            System.err.println("[INIT-APPROX-NNS-TABLE] Set is too small: " + nSamples + " sequence. At least 2 sequences needed.");
        }

        candidateNNS = new CandidateNN[nParams][nSamples];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < nSamples; ++len) {
                candidateNNS[paramId][len] = new CandidateNN();
            }
        }

        final LazyAssessNN_DTW[] lazyAssessNNS = new LazyAssessNN_DTW[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNN_DTW(cache);
        }
        final ArrayList<LazyAssessNN_DTW> challengers = new ArrayList<>(nSamples);

        for (int current = 0; current < nSamples; ++current) {
            final Instance sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < train.size(); ++previous) {
                if (previous == current) continue;

                final LazyAssessNN_DTW d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(train, paramId);
                final int win = getWindowSize(maxWindow);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                Collections.sort(challengers);
                boolean newNN = false;
                for (LazyAssessNN_DTW challenger : challengers) {
                    // --- Get the data
                    int previous = challenger.indexQuery;
                    if (previous == current) previous = challenger.indexReference;
                    if (previous == currPNN.index) continue;

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    LazyAssessNN_DTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win);
                    // --- Check the result
                    if (rrt == LazyAssessNN_DTW.RefineReturnType.New_best) {
                        int r = challenger.getMinWindowValidityForFullDistance();
                        double d = challenger.getDistance(win);
                        currPNN.set(previous, r, d, CandidateNN.Status.BC);
                        newNN = true;
                    }

                    if (previous < nSamples) {
                        CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNN_DTW.RefineReturnType.New_best) {
                            int r = challenger.getMinWindowValidityForFullDistance();
                            double d = challenger.getDistance(win);
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
                        }
                    }
                }
                if (newNN) {
                    int r = currPNN.r;
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    int winEnd = getParamIdFromWindow(r, train.numAttributes() - 1);
                    for (int tmp = paramId; tmp >= winEnd; --tmp) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                    }
                }
            }
        }
    }

    protected int getParamIdFromWindow(final int w, final int n) {
        double r = 1.0 * w / n;
        return (int) Math.ceil(r * 100);
    }

}
