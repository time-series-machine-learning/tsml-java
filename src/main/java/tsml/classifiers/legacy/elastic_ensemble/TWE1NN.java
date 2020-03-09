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
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.assessingNN.LazyAssessNN_TWED;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.lowerBounds.LbTwed;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import machine_learning.classifiers.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.TWEDistance;

import java.util.ArrayList;
import java.util.Collections;
//import efficient_standalone_classifiers.Eff

/**
 * written April '16 - looks good
 *
 * @author sjx07ngu
 */
public class TWE1NN extends Efficient1NN {


    private static final double DEGREE = 2; // not bothering to set the degree in this code, it's fixed to 2 in the other anyway
    double nu = 1;
    double lambda = 1;


    protected static double[] twe_nuParams = {
            // <editor-fold defaultstate="collapsed" desc="hidden for space">
            0.00001,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1,// </editor-fold>
    };

    protected static double[] twe_lamdaParams = {
            // <editor-fold defaultstate="collapsed" desc="hidden for space">
            0,
            0.011111111,
            0.022222222,
            0.033333333,
            0.044444444,
            0.055555556,
            0.066666667,
            0.077777778,
            0.088888889,
            0.1,// </editor-fold>
    };

    public TWE1NN(double nu, double lambda) {
        this.nu = nu;
        this.lambda = lambda;
        this.classifierIdentifier = "TWE_1NN";
        this.allowLoocv = false;
    }

    public TWE1NN() {
        // note: these defaults may be garbage for most measures. Should set them through CV or prior knowledge
        this.nu = 0.005;
        this.lambda = 0.5;
        this.classifierIdentifier = "TWE_1NN";
    }

    public final double distance(Instance first, Instance second, double cutoff) {
        // note: I can't see a simple way to use the cutoff, so unfortunately there isn't one! 

        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if (first.classIndex() != first.numAttributes() - 1 || second.classIndex() != second.numAttributes() - 1) {
            return new TWEDistance(nu, lambda).distance(first, second, cutoff);
        }

        int m = first.numAttributes() - 1;
        int n = second.numAttributes() - 1;

        int dim = 1;
        double dist, disti1, distj1;
        double[][] ta = new double[m][dim];
        double[][] tb = new double[m][dim];
        double[] tsa = new double[m];
        double[] tsb = new double[n];

        // look like time staps

        for (int i = 0; i < tsa.length; i++) {
            tsa[i] = (i + 1);
        }
        for (int i = 0; i < tsb.length; i++) {
            tsb[i] = (i + 1);
        }

        int r = ta.length; // this is just m?!
        int c = tb.length; // so is this, but surely it should actually be n anyway


        int i, j, k;
//Copy over values
        for (i = 0; i < m; i++) {
            ta[i][0] = first.value(i);
        }
        for (i = 0; i < n; i++) {
            tb[i][0] = second.value(i);
        }

        /* allocations in c
         double **D = (double **)calloc(r+1, sizeof(double*));
         double *Di1 = (double *)calloc(r+1, sizeof(double));
         double *Dj1 = (double *)calloc(c+1, sizeof(double));
         for(i=0; i<=r; i++) {
         D[i]=(double *)calloc(c+1, sizeof(double));
         }
         */
        double[][] D = new double[r + 1][c + 1];
        double[] Di1 = new double[r + 1];
        double[] Dj1 = new double[c + 1];
// local costs initializations
        for (j = 1; j <= c; j++) {
            distj1 = 0;
            for (k = 0; k < dim; k++) {
                if (j > 1) {
//CHANGE AJB 8/1/16: Only use power of 2 for speed                      
                    distj1 += (tb[j - 2][k] - tb[j - 1][k]) * (tb[j - 2][k] - tb[j - 1][k]);
// OLD VERSION                    distj1+=Math.pow(Math.abs(tb[j-2][k]-tb[j-1][k]),degree);
// in c:               distj1+=pow(fabs(tb[j-2][k]-tb[j-1][k]),degree);
                } else {
                    distj1 += tb[j - 1][k] * tb[j - 1][k];
                }
            }
//OLD              		distj1+=Math.pow(Math.abs(tb[j-1][k]),degree);
            Dj1[j] = (distj1);
        }

        for (i = 1; i <= r; i++) {
            disti1 = 0;
            for (k = 0; k < dim; k++) {
                if (i > 1) {
                    disti1 += (ta[i - 2][k] - ta[i - 1][k]) * (ta[i - 2][k] - ta[i - 1][k]);
                } // OLD                 disti1+=Math.pow(Math.abs(ta[i-2][k]-ta[i-1][k]),degree);
                else {
                    disti1 += (ta[i - 1][k]) * (ta[i - 1][k]);
                }
            }
//OLD                  disti1+=Math.pow(Math.abs(ta[i-1][k]),degree);

            Di1[i] = (disti1);

            for (j = 1; j <= c; j++) {
                dist = 0;
                for (k = 0; k < dim; k++) {
                    dist += (ta[i - 1][k] - tb[j - 1][k]) * (ta[i - 1][k] - tb[j - 1][k]);
//                  dist+=Math.pow(Math.abs(ta[i-1][k]-tb[j-1][k]),degree);
                    if (i > 1 && j > 1) {
                        dist += (ta[i - 2][k] - tb[j - 2][k]) * (ta[i - 2][k] - tb[j - 2][k]);
                    }
//                    dist+=Math.pow(Math.abs(ta[i-2][k]-tb[j-2][k]),degree);
                }
                D[i][j] = (dist);
            }
        }// for i

        // border of the cost matrix initialization
        D[0][0] = 0;
        for (i = 1; i <= r; i++) {
            D[i][0] = D[i - 1][0] + Di1[i];
        }
        for (j = 1; j <= c; j++) {
            D[0][j] = D[0][j - 1] + Dj1[j];
        }

        double dmin, htrans, dist0;
        int iback;

        for (i = 1; i <= r; i++) {
            for (j = 1; j <= c; j++) {
                htrans = Math.abs((tsa[i - 1] - tsb[j - 1]));
                if (j > 1 && i > 1) {
                    htrans += Math.abs((tsa[i - 2] - tsb[j - 2]));
                }
                dist0 = D[i - 1][j - 1] + nu * htrans + D[i][j];
                dmin = dist0;
                if (i > 1) {
                    htrans = ((tsa[i - 1] - tsa[i - 2]));
                } else {
                    htrans = tsa[i - 1];
                }
                dist = Di1[i] + D[i - 1][j] + lambda + nu * htrans;
                if (dmin > dist) {
                    dmin = dist;
                }
                if (j > 1) {
                    htrans = (tsb[j - 1] - tsb[j - 2]);
                } else {
                    htrans = tsb[j - 1];
                }
                dist = Dj1[j] + D[i][j - 1] + lambda + nu * htrans;
                if (dmin > dist) {
                    dmin = dist;
                }
                D[i][j] = dmin;
            }
        }

        dist = D[r][c];
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

        Instances train = DatasetLoading.loadDataNullable(tscProbDir + datasetName + "/" + datasetName + "_TRAIN");
        Instances test = DatasetLoading.loadDataNullable(tscProbDir + datasetName + "/" + datasetName + "_TEST");

        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        TWEDistance oldDtw = new TWEDistance(0.001, 0.5);
        knn.setDistanceFunction(oldDtw);
        knn.buildClassifier(train);

        // new version
        TWE1NN dtwNew = new TWE1NN(0.001, 0.5);
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
        this.nu = twe_nuParams[paramId / 10];
        this.lambda = twe_lamdaParams[paramId % 10];
    }

    @Override
    public String getParamInformationString() {
        return this.nu + "," + this.lambda;
    }

    /************************************************************************************************
     Support for FastEE
     @author Chang Wei Tan, Monash University (chang.tan@monash.edu)
     ************************************************************************************************/
    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex) {
        return LbTwed.distance(query, candidate, trainCache.getMax(queryIndex), trainCache.getMin(queryIndex), nu, lambda);
    }

    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex, final double cutOffValue) {
        return LbTwed.distance(query, candidate, trainCache.getMax(queryIndex), trainCache.getMin(queryIndex), nu, lambda, cutOffValue);
    }

    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex, final SequenceStatsCache cache) {
        return LbTwed.distance(query, candidate, cache.getMax(queryIndex), cache.getMin(queryIndex), nu, lambda);
    }

    @Override
    public double lowerBound(final Instance query, final Instance candidate, final int queryIndex, final int candidateIndex, final double cutOffValue, final SequenceStatsCache cache) {
        return LbTwed.distance(query, candidate, cache.getMax(queryIndex), cache.getMin(queryIndex), nu, lambda, cutOffValue);
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

        final LazyAssessNN_TWED[] lazuAssessNN = new LazyAssessNN_TWED[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazuAssessNN[i] = new LazyAssessNN_TWED(cache);
        }
        final ArrayList<LazyAssessNN_TWED> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            // --- --- Get the data --- ---
            final Instance sCurrent = train.get(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNN_TWED d = lazuAssessNN[previous];
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
                        // --- Get the data
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNN_TWED challenger = lazuAssessNN[previous];
                        final LazyAssessNN_TWED.RefineReturnType rrt = challenger.tryToBeat(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN_TWED.RefineReturnType.New_best) {
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
                    // We don't have NN yet.
                    // Sort the challengers so we have better chance to organize good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNN_TWED challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNN_TWED.RefineReturnType rrt = challenger.tryToBeat(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN_TWED.RefineReturnType.New_best) {
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
                        challenger = lazuAssessNN[previous];
                        rrt = challenger.tryToBeat(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN_TWED.RefineReturnType.New_best) {
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
                    final double d = currPNN.distance;
                    final int index = currPNN.index;
                    candidateNNS[paramId][current].set(index, d, CandidateNN.Status.NN);
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

        final LazyAssessNN_TWED[] lazyAssessNN = new LazyAssessNN_TWED[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNN[i] = new LazyAssessNN_TWED(cache);
        }
        final ArrayList<LazyAssessNN_TWED> challengers = new ArrayList<>(nSamples);

        for (int current = 0; current < nSamples; ++current) {
            // --- --- Get the data --- ---
            final Instance sCurrent = train.get(current);

            // Clear off the previous challengers and add all the previous sequences
            challengers.clear();
            for (int previous = 0; previous < train.size(); ++previous) {
                if (previous == current) continue;

                final LazyAssessNN_TWED d = lazyAssessNN[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = 0; paramId < nParams; ++paramId) {
                setParamsFromParamId(train, paramId);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                Collections.sort(challengers);
                boolean newNN = false;
                for (LazyAssessNN_TWED challenger : challengers) {
                    // --- Get the data
                    int previous = challenger.indexQuery;
                    if (previous == current) previous = challenger.indexReference;
                    if (previous == currPNN.index) continue;

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    LazyAssessNN_TWED.RefineReturnType rrt = challenger.tryToBeat(toBeat, nu, lambda);

                    // --- Check the result
                    if (rrt == LazyAssessNN_TWED.RefineReturnType.New_best) {
                        double d = challenger.getDistance();
                        currPNN.set(previous, d, CandidateNN.Status.BC);
                        newNN = true;
                    }

                    if (previous < nSamples) {
                        CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNN[previous];
                        rrt = challenger.tryToBeat(toBeat, nu, lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNN_TWED.RefineReturnType.New_best) {
                            double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
                        }
                    }
                }

                if (newNN) {
                    double d = currPNN.distance;
                    int index = currPNN.index;
                    candidateNNS[paramId][current].set(index, d, CandidateNN.Status.NN);
                }
            }
        }
    }


}
