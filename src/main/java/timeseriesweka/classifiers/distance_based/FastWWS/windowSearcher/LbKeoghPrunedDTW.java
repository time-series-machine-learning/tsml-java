/*******************************************************************************
 * Copyright (C) 2017 Chang Wei Tan
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

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;

import timeseriesweka.classifiers.distance_based.FastWWS.items.MonoDoubleItemSet;
import timeseriesweka.classifiers.distance_based.FastWWS.sequences.SymbolicSequence;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * Search for the best warping window using PrunedDTW
 * 
 * We use the original PrunedDTW C++ code from http://sites.labic.icmc.usp.br/prunedDTW/ 
 * and modify it into Java
 * 
 * Original paper: 
 * Silva, D. F., & Batista, G. E. (2016, June). 
 * Speeding up all-pairwise dynamic time warping matrix calculation. 
 * In Proceedings of the 2016 SIAM International Conference on Data Mining (pp. 837-845).
 *  Society for Industrial and Applied Mathematics.
 * 
 * @author Chang Wei Tan
 *
 */
public class LbKeoghPrunedDTW extends WindowSearcher {
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    private static final long serialVersionUID = -1561497612657542978L;
	public PrintStream out;								// Output print
	
	private String[] searchResults;						// Our results
	private int[][] nns;								// Similar to our main structure
	private double[][] dist;							// Matrix to store the distances
	
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	public LbKeoghPrunedDTW() {
		super();
		out = System.out;
	}
	
	public LbKeoghPrunedDTW(String name) {
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
		
		classedData = new HashMap<>();
		classedDataIndices = new HashMap<>();
		for (int c = 0; c < data.numClasses(); c++) {
			classedData.put(data.classAttribute().value(c), new ArrayList<SymbolicSequence>());
			classedDataIndices.put(data.classAttribute().value(c), new ArrayList<Integer>());
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
		searchResults = new String[maxWindow+1];
		nns = new int[maxWindow+1][train.length];
		dist = new double[train.length][train.length];
		
		// Start searching for the best window
		searchBestWarpingWindow();
		
		// Saving best windows found
		System.out.println("Windows found=" + bestWarpingWindow + " Best Acc=" + (1-bestScore));
	}
	
    /**
     * This is similar to buildClassifier but it is an estimate. 
     * This is used for large dataset where it takes very long to run.
     * The main purpose of this is to get the run time and not actually search for the best window.
     * We call this to draw Figure 1 of our SDM18 paper
     * @param data
     * @param estimate
     * @throws Exception
     */
	@Override
	public void buildClassifierEstimate(Instances data, int estimate) throws Exception {
		// Initialise training dataset
		Attribute classAttribute = data.classAttribute();
		
		classedData = new HashMap<>();
		classedDataIndices = new HashMap<>();
		for (int c = 0; c < data.numClasses(); c++) {
			classedData.put(data.classAttribute().value(c), new ArrayList<SymbolicSequence>());
			classedDataIndices.put(data.classAttribute().value(c), new ArrayList<Integer>());
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
		searchResults = new String[maxWindow+1];
		nns = new int[maxWindow+1][train.length];
		dist = new double[train.length][train.length];
		
		int[] nErrors = new int[maxWindow+1];
		double[] score = new double[maxWindow+1];
		double bestScore = Double.MAX_VALUE;
		double minD;
		bestWarpingWindow=-1;

		// Start searching for the best window.
		// Only loop through a given size of the dataset, but still search for NN from the whole train
		// for every sequence in train, we find NN for all window
		// then in the end, update the best score
		for (int i = 0; i < estimate; i++) {
			SymbolicSequence testSeq = train[i];

			for (int w = 0; w <= maxWindow; w++){
				testSeq.LB_KeoghFillUL(w, U, L);
				
				minD = Double.MAX_VALUE;
				String classValue = null;
				for (int j = 0; j < train.length; j++) {
					if (i == j)
						continue;
					SymbolicSequence trainSeq = train[j];

					if (SymbolicSequence.LB_KeoghPreFilled(trainSeq, U, L) < minD) {
						double tmpD;
						if (w == 0) {
							tmpD = testSeq.PrunedDTW(trainSeq, w);
						} else {
							tmpD = testSeq.PrunedDTW(trainSeq, w, dist[i][j]);
						}
						if (tmpD < minD) {
							minD = tmpD;
							classValue = classMap[j];
							nns[w][i] = j;
						}
						dist[i][j] = tmpD*tmpD;
					} else {
						if (w > 0) {
							dist[i][j] = dist[i][j];
						} else {
							dist[i][j] = Double.MAX_VALUE;
						}
					}
				}
				if (classValue == null || !classValue.equals(classMap[i])) {
					nErrors[w]++;
				}
				score[w] = 1.0 * nErrors[w]/train.length;
			}
		}
		
		for (int w = 0; w < maxWindow; w++) {
			if (score[w] < bestScore) {
				bestScore = score[w];
				bestWarpingWindow = w;
			}
		}
		
		// Saving best windows found
		System.out.println("Windows found=" + bestWarpingWindow + " Best Acc=" + (1-bestScore));
	}
	
	/**
	 * Search for the best warping window 
	 * for every window, we evaluate the performance of the classifier
	 */
	@Override
	protected void searchBestWarpingWindow(){
		int currentWindow = 0;
		double currentScore = 1.0;
		bestScore = 1.0;
		
		long startTime = System.currentTimeMillis();
		
		// Start from smallest window, w=0
		while (currentWindow >= 0 && currentWindow<= maxWindow) {
			
			currentScore = evalSolution(currentWindow);
			
			long endTime = System.currentTimeMillis();
			long accumulatedTime = (endTime-startTime);
			
			// saving results
			searchResults[currentWindow] = currentWindow + "," + currentScore + "," + accumulatedTime;
			
//			out.println(currentWindow+" "+currentScore+" "+accumulatedTime);
//			out.flush();
			
			if (currentScore < bestScore) {
				bestScore = currentScore;
				bestWarpingWindow = currentWindow;
			}
			
			currentWindow = currentWindow + 1;
		}
	}
	
	/**
	 * Evaluate the performance of the classifier 
	 * Here we use LB Keogh to further speed up the process
	 * Original code does not have LB Keogh
	 * 
	 * @param warpingWindow
	 * @return
	 */
	@Override
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
					double tmpD;
					if (warpingWindow == 0) {
						tmpD = testSeq.PrunedDTW(trainSeq, warpingWindow);
					} else {
						tmpD = testSeq.PrunedDTW(trainSeq, warpingWindow, dist[i][j]);
					}
					if (tmpD < minD) {
						minD = tmpD;
						classValue = classMap[j];
						nns[warpingWindow][i] = j;
					}
					dist[i][j] = tmpD*tmpD;
				} else {
					if (warpingWindow > 0) {
						dist[i][j] = dist[i][j];
					} else {
						dist[i][j] = Double.MAX_VALUE;
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
				double tmpD = seq.DTW(s,bestWarpingWindow, warpingMatrix);
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
	 * Get our search results
	 * @return
	 */
	@Override
	public String[] getSearchResults() {
		return searchResults;
	}

	/** 
	 * Get the best warping window found
	 */
	@Override
	public int getBestWin() {
		return bestWarpingWindow;
	}

	/** 
	 * Get the LOOCV accuracy for the best warping window
	 */
	@Override
	public double getBestScore() {
		return bestScore;
	}

	/** 
	 * Set the result directory
	 */
	@Override
	public void setResDir(String path) {
		resDir = path;
	}

	/** 
	 * Set type of classifier
	 */
	@Override
	public void setType(String t) {
		type = t;
	}
}
