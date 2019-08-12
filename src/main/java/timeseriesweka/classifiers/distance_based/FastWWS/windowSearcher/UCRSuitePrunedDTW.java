/*******************************************************************************
 * Copyright (C) 2017 Chang Wei Tan
 * 
 * This file is part of FastWWSearch.
 * 
 * FastWWSearch is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * TSI is distributed in the hope that it will be useful,
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

import timeseriesweka.classifiers.distance_based.FastWWS.items.LazyAssessNNEarlyAbandon;
import timeseriesweka.classifiers.distance_based.FastWWS.items.LazyAssessNNEarlyAbandon.RefineReturnType;
import timeseriesweka.classifiers.distance_based.FastWWS.sequences.SymbolicSequence;
import timeseriesweka.classifiers.distance_based.FastWWS.items.MonoDoubleItemSet;
import timeseriesweka.classifiers.distance_based.FastWWS.items.SequenceStatsCache;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * Search for the best warping window using Cascading Lower Bound and Early Abandon with PrunedDTW
 * Replace the standard DTW with PrunedDTW
 * 
 * Original papers:
 * Rakthanmanon, T., Campana, B., Mueen, A., Batista, G., Westover, B., Zhu, Q., ... & Keogh, E. (2012, August). 
 * Searching and mining trillions of time series subsequences under dynamic time warping. 
 * In Proceedings of the 18th ACM SIGKDD international conference 
 * on Knowledge discovery and data mining (pp. 262-270). ACM. Chicago
 * 
 * Silva, D. F., & Batista, G. E. (2016, June). 
 * Speeding up all-pairwise dynamic time warping matrix calculation. 
 * In Proceedings of the 2016 SIAM International Conference on Data Mining (pp. 837-845).
 *  Society for Industrial and Applied Mathematics.
 *  
 * @author Chang Wei Tan
 *
 */
public class UCRSuitePrunedDTW extends WindowSearcher {
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	private static final long serialVersionUID = -1561497612657542978L;
	public PrintStream out;							// Output print

	private String[] searchResults;					// Our results
	private int[][] nns;							// Similar to our main structure
	private double[][] dist;						// Matrix to stroe the distances
	
	private SequenceStatsCache cache;				// Cache to store the information for the sequences
	
	private LazyAssessNNEarlyAbandon[][] lazyUCR;		// LazyUCR distance with Early Abandon
	
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	public UCRSuitePrunedDTW() {
		super();
		out = System.out;
	}
	
	public UCRSuitePrunedDTW(String name) {
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
		U1 = new double[maxLength];
		L1 = new double[maxLength];
		
		maxWindow = Math.round(1 * maxLength);
		searchResults = new String[maxWindow+1];
		nns = new int[maxWindow+1][train.length];
		dist = new double[train.length][train.length];

		cache = new SequenceStatsCache(train, maxWindow);
		
		lazyUCR = new LazyAssessNNEarlyAbandon[train.length][train.length];
		
		for (int i = 0; i < train.length; i++) {
			for (int j  = 0; j < train.length; j++) {
				lazyUCR[i][j] = new LazyAssessNNEarlyAbandon(cache);
			}
		}
		
		// Start searching for the best window
		searchBestWarpingWindow();
	
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
	 * 
	 * @param warpingWindow
	 * @return
	 */
	@Override
	protected double evalSolution(int warpingWindow) {
		int nErrors = 0;
		// test fold number is nFold
		for (int i = 0; i < train.length; i++) {
			double minD = Double.MAX_VALUE;
			String classValue = null;
			for (int j = 0; j < train.length; j++) {
				if (i == j)
					continue;
				
				lazyUCR[i][j].set(train[i], i, train[j], j);
				RefineReturnType rrt;
				if (warpingWindow == 0) 
					rrt = lazyUCR[i][j].tryToBeatPrunedDTW(minD, warpingWindow);
				else
					rrt = lazyUCR[i][j].tryToBeatPrunedDTW(minD, warpingWindow, dist[i][j]);
				
				if (rrt == RefineReturnType.New_best) {
					minD = lazyUCR[i][j].getDistance(warpingWindow);
					nns[warpingWindow][i] = j;
					classValue = classMap[j];
					dist[i][j] = minD;
				} else if (rrt == RefineReturnType.Pruned_with_DTW) {
					dist[i][j] = lazyUCR[i][j].getDistance(warpingWindow);
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
