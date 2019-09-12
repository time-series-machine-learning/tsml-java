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
import timeseriesweka.classifiers.distance_based.FastWWS.items.SequenceStatsCache;
import timeseriesweka.classifiers.distance_based.FastWWS.sequences.SymbolicSequence;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 *
 * Search for the best warping window using Cascading Lower Bound and Early Abandon
 * 1. LB Kim
 * 2. LB Keogh (Q,C) EA
 * 3. LB Keogh (C,Q) EA
 * 4. DTW
 *
 * Original paper:
 * Rakthanmanon, T., Campana, B., Mueen, A., Batista, G., Westover, B., Zhu, Q., ... & Keogh, E. (2012, August).
 * Searching and mining trillions of time series subsequences under dynamic time warping.
 * In Proceedings of the 18th ACM SIGKDD international conference
 * on Knowledge discovery and data mining (pp. 262-270). ACM. Chicago
 *
 * @author Chang Wei Tan
 *
 */
public class Trillion extends WindowSearcher {
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	private static final long serialVersionUID = 1L;
	public PrintStream out;						// Output print

	private SequenceStatsCache cache;			// Cache to store the information for the sequences

	private SymbolicSequence query, reference;	// Query and reference sequences
	private int indexQuery, indexReference;		// Index for query and reference
	private double minDist, bestMinDist;		// Distance and best so far distance
	private int currentW;						// Current warping window
	private int indexStoppedLB;					// Index where we stop LB

	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	public Trillion() {
		super();
		out = System.out;
	}

	public Trillion(String name) {
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

		U = new double[maxLength];
		L = new double[maxLength];
		maxWindow = Math.round(1 * maxLength);
		cache = new SequenceStatsCache(train, maxWindow);

		int nbErrors = 0;
		double score;
		bestScore = Double.MAX_VALUE;
		bestWarpingWindow=-1;

		// Start searching for the best window
		for (int w = 0; w <= maxWindow; w++) {
			currentW = w;
			nbErrors = 0;
			for (int i = 0; i < train.length; i++) {
				query = train[i];
				indexQuery = i;
				bestMinDist = Double.MAX_VALUE;
				String classValue = null;
				for (int j = 0; j < train.length; j++) {
					if (i==j)
						continue;
					reference = train[j];
					indexReference = j;

					// LB Kim
					doLBKim();
					if (minDist < bestMinDist) {
						minDist = 0;
						indexStoppedLB = 0;
						// LB Keogh(Q,R)
						doLBKeoghQR(bestMinDist);
						if (minDist < bestMinDist) {
							minDist = 0;
							indexStoppedLB = 0;
							// LB Keogh(R,Q)
							doLBKeoghRQ(bestMinDist);
							if (minDist < bestMinDist) {
								// DTW
								double res = query.DTW(reference, currentW);
								minDist = res * res;
								if(minDist < bestMinDist){
									bestMinDist = minDist;
									classValue = classMap[j];
								}
							}
						}
					}
				}
				if (classValue == null || !classValue.equals(classMap[i])) {
					nbErrors++;
				}
			}
			score = 1.0 * nbErrors / train.length;
			if (score < bestScore) {
				bestScore = score;
				bestWarpingWindow = w;
			}
		}

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

		U = new double[maxLength];
		L = new double[maxLength];
		maxWindow = Math.round(1 * maxLength);
		cache = new SequenceStatsCache(train, maxWindow);

		int[] nbErrors = new int[maxWindow+1];
		double[] score = new double[maxWindow+1];
		bestScore = Double.MAX_VALUE;
		bestWarpingWindow=-1;

		// Start searching for the best window.
		// Only loop through a given size of the dataset, but still search for NN from the whole train
		// for every sequence in train, we find NN for all window
		// then in the end, update the best score
		for (int i = 0; i < estimate; i++) {
			query = train[i];
			indexQuery = i;

			for (int w = 0; w <= maxWindow; w++) {
				currentW = w;
				bestMinDist = Double.MAX_VALUE;
				String classValue = null;
				for (int j = 0; j < train.length; j++) {
					if (i==j)
						continue;
					reference = train[j];
					indexReference = j;

					// LB Kim
					doLBKim();
					if (minDist < bestMinDist) {
						minDist = 0;
						indexStoppedLB = 0;
						// LB Keogh(Q,R)
						doLBKeoghQR(bestMinDist);
						if (minDist < bestMinDist) {
							minDist = 0;
							indexStoppedLB = 0;
							// LB Keogh(R,Q)
							doLBKeoghRQ(bestMinDist);
							if (minDist < bestMinDist) {
								// DTW
								double res = query.DTW(reference, currentW);
								minDist = res * res;
								if(minDist < bestMinDist){
									bestMinDist = minDist;
									classValue = classMap[j];
								}
							}
						}
					}
				}
				if (classValue == null || !classValue.equals(classMap[i])) {
					nbErrors[w]++;
				}
				score[w] = 1.0 * nbErrors[w]/train.length;
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
				double tmpD = seq.DTW(s,bestWarpingWindow);
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
	 * Run LB Kim using data from cache
	 */
	public void doLBKim() {
		double diffFirsts = query.sequence[0].squaredDistance(reference.sequence[0]);
		double diffLasts = query.sequence[query.getNbTuples() - 1].squaredDistance(reference.sequence[reference.getNbTuples() - 1]);
		minDist = diffFirsts + diffLasts;
		if(!cache.isMinFirst(indexQuery)&&!cache.isMinFirst(indexReference) && !cache.isMinLast(indexQuery) && !cache.isMinLast(indexReference)){
			double diffMin = cache.getMin(indexQuery)-cache.getMin(indexReference);
			minDist += diffMin*diffMin;
		}
		if(!cache.isMaxFirst(indexQuery)&&!cache.isMaxFirst(indexReference)&& !cache.isMaxLast(indexQuery) && !cache.isMaxLast(indexReference)){
			double diffMax = cache.getMax(indexQuery)-cache.getMax(indexReference);
			minDist += diffMax*diffMax;
		}
	}

	/**
	 * Run LB Keogh(Q,R) with EA using data from cache
	 * @param scoreToBeat
	 */
	public void doLBKeoghQR(double scoreToBeat) {
		int length = query.sequence.length;
		double[] LEQ = cache.getLE(indexQuery, currentW);
		double[] UEQ = cache.getUE(indexQuery, currentW);
		while (indexStoppedLB < length && minDist < scoreToBeat) {
			int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
			double c = ((MonoDoubleItemSet) reference.sequence[index]).value;
			if (c < LEQ[index]) {
				double diff = LEQ[index] - c;
				minDist += diff * diff;
			} else if (UEQ[index] < c) {
				double diff = UEQ[index] - c;
				minDist += diff * diff;
			}
			indexStoppedLB++;
		}
	}

	/**
	 * Run LB Keogh(R,Q) with EA using data from cache
	 * @param scoreToBeat
	 */
	public void doLBKeoghRQ(double scoreToBeat) {
		int length = reference.sequence.length;
		double[] LER = cache.getLE(indexReference, currentW);
		double[] UER = cache.getUE(indexReference, currentW);
		while (indexStoppedLB < length && minDist < scoreToBeat) {
			int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
			double c = ((MonoDoubleItemSet) query.sequence[index]).value;
			if (c < LER[index]) {
				double diff = LER[index] - c;
				minDist += diff * diff;
			} else if (UER[index] < c) {
				double diff = UER[index] - c;
				minDist += diff * diff;
			}
			indexStoppedLB++;
		}
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
