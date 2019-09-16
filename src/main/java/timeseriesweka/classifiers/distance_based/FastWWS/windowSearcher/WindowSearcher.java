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
package timeseriesweka.classifiers.distance_based.FastWWS.windowSearcher;

import timeseriesweka.classifiers.distance_based.FastWWS.items.MonoDoubleItemSet;
import timeseriesweka.classifiers.distance_based.FastWWS.sequences.SymbolicSequence;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import weka.classifiers.AbstractClassifier;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * <p>
 * Superclass for all the classifiers that we used
 * By default, it uses DTW with LB Keogh to search for the best window using Algorithm 2 in our SDM18 paper
 * In our SDM18 paper, we call this as the Naive method
 * Refer to http://www.cs.ucr.edu/~eamonn/LB_Keogh.htm for the details of LB Keogh
 *
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 */
public class WindowSearcher extends AbstractClassifier{
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    private static final long serialVersionUID = -1561497612657542978L;
    protected static int bestWarpingWindow;                                    // Best warping window found
    protected static int bestWindowPercent = -1;                               // Best warping window found in percentage
    protected static double bestScore;                                        // Best LOOCV accuracy for the best warping window
    protected static String type = "keogh";                                    // Default type is DTW with LB Keogh
    protected static String datasetName;                                    // Name of dataset that is being tested
    protected static String resDir = "/home/changwei/workspace/FindBestWarpingWindow/outputs/";    // result directory
    public PrintStream out;                                                    // Output print
    protected boolean forwardSearch = false;                                    // Search from front or back
    protected boolean greedySearch = false;                                    // Greedy or not
    protected SymbolicSequence[] train;                                        // Training dataset, array of sequences
    protected HashMap <String, ArrayList <SymbolicSequence>> classedData;        // Sequences by classes
    protected HashMap <String, ArrayList <Integer>> classedDataIndices;        // Sequences index in train
    protected String[] classMap;                                            // Class per index
    protected double[][] warpingMatrix;                                        // DTW cost matrix
    protected double[] U, L, U1, L1;                                        // Upper and lower envelope for LB Keogh
    protected int maxLength, maxWindow;                                        // Max length of the sequences
    protected String[] searchResults;                                        // Our results
    protected int nParams = 100;      // this can be the maximum length or percentage
    private int[][] nns;                                                    // Similar to our main structure
    private double[][] dist;                                                // Matrix to store the distances

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    public WindowSearcher() {
        super();
        out = System.out;
    }

    public WindowSearcher(String name) {
        super();
        out = System.out;
        datasetName = name;
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Methods
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    @Override
    public void buildClassifier(Instances data) throws Exception {
        // Initialise training dataset
        Attribute classAttribute = data.classAttribute();

        classedData = new HashMap <>();
        classedDataIndices = new HashMap <>();
        for (int c = 0; c < data.numClasses(); c++) {
            classedData.put(data.classAttribute().value(c), new ArrayList <SymbolicSequence>());
            classedDataIndices.put(data.classAttribute().value(c), new ArrayList <Integer>());
        }

        train = new SymbolicSequence[data.numInstances()];
        classMap = new String[train.length];
        maxLength = 0;
        for (int i = 0; i < train.length; i++) {
            Instance sample = data.instance(i);
            MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
            maxLength = Math.max(maxLength, sequence.length);
            int shift = (sample.classIndex() == 0) ? 1 : 0;
            for (int t = 0; t < sequence.length; t++) {
                sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
            }
            train[i] = new SymbolicSequence(sequence);
            String clas = sample.stringValue(classAttribute);
            classMap[i] = clas;
            classedData.get(clas).add(train[i]);
            classedDataIndices.get(clas).add(i);
        }

        warpingMatrix = new double[maxLength][maxLength];
        U = new double[maxLength];
        L = new double[maxLength];

        maxWindow = Math.round(1 * maxLength);
        nns = new int[maxWindow + 1][train.length];
        dist = new double[maxWindow + 1][train.length];

        // Start searching for the best window
        searchBestWarpingWindow();

        // if we are doing length, find the best window in percentage
        if (bestWindowPercent < 0)
            bestWindowPercent = lengthToPercent(bestWarpingWindow);

        // Saving best windows found
        System.out.println("Windows found=" + bestWarpingWindow +
                "(" + bestWindowPercent + ") Best Acc=" + (1 - bestScore));
    }

    /**
     * This is similar to buildClassifier but it is an estimate.
     * This is used for large dataset where it takes very long to run.
     * The main purpose of this is to get the run time and not actually search for the best window.
     * We use this to draw Figure 1 of our SDM18 paper
     *
     * @param data
     * @param estimate
     * @throws Exception
     */
    public void buildClassifierEstimate(Instances data, int estimate) throws Exception {
        // Initialise training dataset
        Attribute classAttribute = data.classAttribute();

        classedData = new HashMap <>();
        classedDataIndices = new HashMap <>();
        for (int c = 0; c < data.numClasses(); c++) {
            classedData.put(data.classAttribute().value(c), new ArrayList <SymbolicSequence>());
            classedDataIndices.put(data.classAttribute().value(c), new ArrayList <Integer>());
        }

        train = new SymbolicSequence[data.numInstances()];
        classMap = new String[train.length];
        maxLength = 0;
        for (int i = 0; i < train.length; i++) {
            Instance sample = data.instance(i);
            MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
            maxLength = Math.max(maxLength, sequence.length);
            int shift = (sample.classIndex() == 0) ? 1 : 0;
            for (int t = 0; t < sequence.length; t++) {
                sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
            }
            train[i] = new SymbolicSequence(sequence);
            String clas = sample.stringValue(classAttribute);
            classMap[i] = clas;
            classedData.get(clas).add(train[i]);
            classedDataIndices.get(clas).add(i);
        }

        warpingMatrix = new double[maxLength][maxLength];
        U = new double[maxLength];
        L = new double[maxLength];

        maxWindow = Math.round(1 * maxLength);
        searchResults = new String[maxWindow + 1];
        nns = new int[maxWindow + 1][train.length];
        dist = new double[maxWindow + 1][train.length];

        int[] nErrors = new int[maxWindow + 1];
        double[] score = new double[maxWindow + 1];
        double bestScore = Double.MAX_VALUE;
        double minD;
        bestWarpingWindow = -1;

        // Start searching for the best window.
        // Only loop through a given size of the dataset, but still search for NN from the whole train
        // for every sequence in train, we find NN for all window
        // then in the end, update the best score
        for (int i = 0; i < estimate; i++) {
            SymbolicSequence testSeq = train[i];

            for (int w = 0; w <= maxWindow; w++) {
                testSeq.LB_KeoghFillUL(w, U, L);

                minD = Double.MAX_VALUE;
                String classValue = null;
                for (int j = 0; j < train.length; j++) {
                    if (i == j)
                        continue;
                    SymbolicSequence trainSeq = train[j];
                    if (SymbolicSequence.LB_KeoghPreFilled(trainSeq, U, L) < minD) {
                        double tmpD = testSeq.DTW(trainSeq, w, warpingMatrix);
                        if (tmpD < minD) {
                            minD = tmpD;
                            classValue = classMap[j];
                            nns[w][i] = j;
                        }
                        dist[w][j] = tmpD * tmpD;
                    }
                }
                if (classValue == null || !classValue.equals(classMap[i])) {
                    nErrors[w]++;
                }
                score[w] = 1.0 * nErrors[w] / train.length;
            }
        }

        for (int w = 0; w < maxWindow; w++) {
            if (score[w] < bestScore) {
                bestScore = score[w];
                bestWarpingWindow = w;
            }
        }

        // Saving best windows found
        System.out.println("Windows found=" + bestWarpingWindow + " Best Acc=" + (1 - bestScore));
    }

    /**
     * Search for the best warping window
     * for every window, we evaluate the performance of the classifier
     */
    protected void searchBestWarpingWindow() {
        int currentWindow = (forwardSearch) ? 0 : maxWindow;
        double currentScore;
        bestScore = 1.0;
        searchResults = new String[maxWindow + 1];

        long startTime = System.currentTimeMillis();

        while (currentWindow >= 0 && currentWindow <= maxWindow) {
            currentScore = evalSolution(currentWindow);

            long endTime = System.currentTimeMillis();
            long accumulatedTime = (endTime - startTime);

            // saving results
            searchResults[currentWindow] = currentWindow + "," + currentScore + "," + accumulatedTime;

//			out.println(currentWindow+" "+currentScore+" "+accumulatedTime);
//			out.flush();

            if (currentScore < bestScore || (currentScore == bestScore && !forwardSearch)) {
                bestScore = currentScore;
                bestWarpingWindow = currentWindow;
            } else if (greedySearch && currentScore > bestScore) {
                break;
            }

            currentWindow = (forwardSearch) ? currentWindow + 1 : currentWindow - 1;
        }
    }

    /**
     * Evaluate the performance of the classifier
     *
     * @param warpingWindow
     * @return
     */
    protected double evalSolution(int warpingWindow) {
        int nErrors = 0;
        // test fold number is nFold
        for (int i = 0; i < train.length; i++) {
            SymbolicSequence testSeq = train[i];
            testSeq.LB_KeoghFillUL(warpingWindow, U, L);

            double minD = Double.MAX_VALUE;
            String classValue = null;
            for (int j = 0; j < train.length; j++) {
                if (i == j)
                    continue;
                SymbolicSequence trainSeq = train[j];

                if (SymbolicSequence.LB_KeoghPreFilled(trainSeq, U, L) < minD) {
                    double tmpD = testSeq.DTW(trainSeq, warpingWindow, warpingMatrix);
                    if (tmpD < minD) {
                        minD = tmpD;
                        classValue = classMap[j];
                        nns[warpingWindow][i] = j;
                        dist[warpingWindow][i] = minD;
                    }
                }
            }

            if (classValue == null || !classValue.equals(classMap[i])) {
                nErrors++;
            }
        }

        return 1.0 * nErrors / train.length;
    }

    @Override
    public double classifyInstance(Instance sample) throws Exception {
        // transform instance to sequence
        MonoDoubleItemSet[] sequence = new MonoDoubleItemSet[sample.numAttributes() - 1];
        int shift = (sample.classIndex() == 0) ? 1 : 0;
        for (int t = 0; t < sequence.length; t++) {
            sequence[t] = new MonoDoubleItemSet(sample.value(t + shift));
        }
        SymbolicSequence seq = new SymbolicSequence(sequence);

        double minD = Double.MAX_VALUE;
        String classValue = null;
        seq.LB_KeoghFillUL(bestWarpingWindow, U, L);

        for (int i = 0; i < train.length; i++) {
            SymbolicSequence s = train[i];
            if (SymbolicSequence.LB_KeoghPreFilled(s, U, L) < minD) {
                double tmpD = seq.DTW(s, bestWarpingWindow, warpingMatrix);
                if (tmpD < minD) {
                    minD = tmpD;
                    classValue = classMap[i];
                }
            }
        }
        // System.out.println(prototypes.size());
        return sample.classAttribute().indexOfValue(classValue);
    }

    /**
     * Convert window in percentage to length
     */
    int percentToLength(int windowInPercent) {
        return maxWindow * windowInPercent / 100;
    }

    /**
     * Convert window in length to percentage
     */
    int lengthToPercent(int windowInLength) {
        double r = 1.0 * windowInLength / maxWindow;
        return (int) Math.ceil(r * 100);
    }

    /**
     * Get our search results
     */
    public String[] getSearchResults() {
        return searchResults;
    }

    /**
     * Get the best warping window found
     */
    public int getBestWin() {
        return bestWarpingWindow;
    }

    /**
     * Get the best warping window in percentage
     */
    public int getBestPercent() {
        return bestWindowPercent;
    }

    /**
     * Get the LOOCV accuracy for the best warping window
     */
    public double getBestScore() {
        return bestScore;
    }

    /**
     * Set the result directory
     */
    public void setResDir(String path) {
        resDir = path;
    }

    /**
     * Set type of classifier
     */
    public void setType(String t) {
        type = t;
    }

    /**
     * Set number of parameters
     * Either maximum length or percentage
     */
    public void setnParams(int n) {
        nParams = n;
    }
}