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
 * Search for the best warping window using just DTW without any optimisation
 * Expect to have the worst runtime!
 * 
 * @author Chang Wei Tan
 *
 */
public class NaiveDTW extends WindowSearcher {
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	private static final long serialVersionUID = -1561497612657542978L;
	public boolean forwardSearch = false;		// Search from front or back
	public boolean greedySearch = false;		// Greedy or not
	public PrintStream out;						// Output print
	
	private String[] searchResults;				// Our results
	private int[][] nns;						// Similar to our main structure
	private double[][] dist;					// Matrix to store the distances
	
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	public NaiveDTW() {
		super();
		out = System.out;
	}
	
	public NaiveDTW(String name) {
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
		dist = new double[maxWindow+1][train.length];
		
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
		int currentWindow = (forwardSearch) ? 0 : maxWindow;
		double currentScore = 1.0;
		bestScore = 1.0;
		
		long startTime = System.currentTimeMillis();
		
		while (currentWindow >= 0 && currentWindow<= maxWindow) {
			currentScore = evalSolution(currentWindow);
			
			long endTime = System.currentTimeMillis();
			long accumulatedTime = (endTime-startTime);
			
			// saving results
			searchResults[currentWindow] = currentWindow + "," + currentScore + "," + accumulatedTime;

			if (currentScore < bestScore || (currentScore == bestScore && !forwardSearch) ) {
				bestScore = currentScore;
				bestWarpingWindow = currentWindow;
			}else if(greedySearch && currentScore > bestScore){
				break;
			}
			
			currentWindow = (forwardSearch) ? currentWindow + 1 : currentWindow - 1;
		}
	}
	
	/**
	 * Evaluate the performance of the classifier 
	 * @param warpingWindow
	 * @return
	 */
	@Override
	protected double evalSolution(int warpingWindow) {
		int nErrors = 0;
		// test fold number is nFold
		for (int i = 0; i < train.length; i++) {
			SymbolicSequence testSeq = train[i];
			
			double minD = Double.MAX_VALUE;
			String classValue = null;
			for (int j = 0; j < train.length; j++) {
				if (i == j)
					continue;
				SymbolicSequence trainSeq = train[j];

				double tmpD = testSeq.DTW(trainSeq, warpingWindow, warpingMatrix);
				if (tmpD < minD) {
					minD = tmpD;
					classValue = classMap[j];
					nns[warpingWindow][i] = j;
					dist[warpingWindow][i] = minD;
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
