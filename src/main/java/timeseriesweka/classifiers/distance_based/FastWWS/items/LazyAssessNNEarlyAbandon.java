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
package timeseriesweka.classifiers.distance_based.FastWWS.items;

import timeseriesweka.classifiers.distance_based.FastWWS.sequences.SymbolicSequence;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * Class for LazyAssessNN distance introduced in our SDM18 paper. 
 * It implements a "lazy" UCR Suites for our KDD12 competitor
 * It is used in CascadeLB.java replacing the original KDD12 code 
 * 
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 *
 */
public class LazyAssessNNEarlyAbandon implements Comparable<LazyAssessNNEarlyAbandon> {
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Internal types
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	public enum RefineReturnType {
		Pruned_with_LB, Pruned_with_DTW, New_best
	}

	public enum LBStatus {
		LB_Kim, 
		Partial_LB_KeoghQR, Full_LB_KeoghQR, 
		Partial_LB_KeoghRQ, Full_LB_KeoghRQ, 
		Previous_Window_LB, Previous_Window_DTW,
		Full_DTW
	}
	
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	protected final static int RIEN = -1;
	protected final static int DIAGONALE = 0;
	protected final static int GAUCHE = 1;
	protected final static int HAUT = 2;
	
	SequenceStatsCache cache;										// Cache to store the information for the sequences
	SymbolicSequence query, reference;								// Query and reference sequences
	public int indexQuery, indexReference;							// Index for query and reference
	int indexStoppedLB, oldIndexStoppedLB;							// Index where we stop LB	
	int currentW;													// Current warping window
	int minWindowValidityFullDTW;									// Minimum window validity for DTW
	int nOperationsLBKim;											// Number of operations for LB Kim
	
	double minDist,LBKeogh1,LBKeogh2,bestMinDist,EuclideanDist;		// Distances
	LBStatus status;												// Status of Lower Bound
		
	public static double[] ubPartials;								// Partial Upper Bound for PrunedDTW

	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	public LazyAssessNNEarlyAbandon(SymbolicSequence query, int index, SymbolicSequence reference, int indexReference, SequenceStatsCache cache) {
		if (index < indexReference) {
			this.query = query;
			this.indexQuery = index;
			this.reference = reference;
			this.indexReference = indexReference;
		} else {
			this.query = reference;
			this.indexQuery = indexReference;
			this.reference = query;
			this.indexReference = index;
		}
		this.minDist = 0.0;
		this.cache = cache;
		tryLBKim();
		this.bestMinDist= minDist;
		this.status = LBStatus.LB_Kim;
	}

	public LazyAssessNNEarlyAbandon(SequenceStatsCache cache){
		this.cache = cache;
	}

	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Method
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	/**
	 * Initialise the distance between query and reference
	 * Reset all parameters
	 * Compute LB Kim 
	 * @param query
	 * @param index
	 * @param reference
	 * @param indexReference
	 */
	public void set (SymbolicSequence query, int index, SymbolicSequence reference, int indexReference) {
		// --- OTHER RESET
		indexStoppedLB = oldIndexStoppedLB = 0;
		currentW = 0;
		minWindowValidityFullDTW = 0;
		nOperationsLBKim = 0;
		LBKeogh1 = LBKeogh2 = 0;
		// --- From constructor
		if (index < indexReference) {
			this.query = query;
			this.indexQuery = index;
			this.reference = reference;
			this.indexReference = indexReference;
		} else {
			this.query = reference;
			this.indexQuery = indexReference;
			this.reference = query;
			this.indexReference = index;
		}
		this.minDist = 0.0;
		tryLBKim();
		this.bestMinDist = minDist;
		this.status = LBStatus.LB_Kim;

	}

	/**
	 * Initialise Upper Bound array for PrunedDTW
	 */
	public void setUBPartial() {
		ubPartials = new double[query.getNbTuples()+1];
	}

	/**
	 * Set the best minimum distance 
	 * @param bestMinDist
	 */
	public void setBestMinDist(double bestMinDist) {
		this.bestMinDist = bestMinDist;
	}

	/**
	 * Set current warping window
	 * @param currentW
	 */
	public void setCurrentW(int currentW) {
		if (this.currentW != currentW) {
			this.currentW = currentW;
			if (status == LBStatus.Full_DTW){
				if(this.currentW >= minWindowValidityFullDTW) {
					this.status = LBStatus.Full_DTW;
				}else{
					this.status = LBStatus.Previous_Window_DTW;
				}
			} else {
				this.status = LBStatus.Previous_Window_LB;
				this.oldIndexStoppedLB = indexStoppedLB;
			}
		}
	}
	
	/** 
	 * Compute Euclidean Distance as Upper Bound for PrunedDTW
	 * @param scoreToBeat
	 * @return
	 */
	public RefineReturnType tryEuclidean(double scoreToBeat) {
		if(bestMinDist>=scoreToBeat){
			return RefineReturnType.Pruned_with_LB;
		}
		if(EuclideanDist >= scoreToBeat) {
			return RefineReturnType.Pruned_with_DTW;
		}
		ubPartials[query.getNbTuples()] = 0;
		for (int i = query.getNbTuples()-1; i >= 0; i--) {
			ubPartials[i] = ubPartials[i+1] + query.getItem(i).squaredDistance(reference.getItem(i));
		}
		EuclideanDist = ubPartials[0];
		return RefineReturnType.New_best;
	}

	/**
	 * Run LB Kim using data from cache
	 */
	protected void tryLBKim() {
		double diffFirsts = query.sequence[0].squaredDistance(reference.sequence[0]);
		double diffLasts = query.sequence[query.getNbTuples() - 1].squaredDistance(reference.sequence[reference.getNbTuples() - 1]);
		minDist = diffFirsts + diffLasts;
		nOperationsLBKim = 2;
		if(!cache.isMinFirst(indexQuery)&&!cache.isMinFirst(indexReference) && !cache.isMinLast(indexQuery) && !cache.isMinLast(indexReference)){
			double diffMin = cache.getMin(indexQuery)-cache.getMin(indexReference);
			minDist += diffMin*diffMin;
			nOperationsLBKim++;
		}
		if(!cache.isMaxFirst(indexQuery)&&!cache.isMaxFirst(indexReference)&& !cache.isMaxLast(indexQuery) && !cache.isMaxLast(indexReference)){
			double diffMax = cache.getMax(indexQuery)-cache.getMax(indexReference);
			minDist += diffMax*diffMax;
			nOperationsLBKim++;
		}
		
		status = LBStatus.LB_Kim;
	}

	/**
	 * Run Full LB Keogh(Q,R) with EA using data from cache
	 */
	protected void tryContinueLBKeoghQR(double scoreToBeat) {
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
	 * Run Full LB Keogh(R,Q) with EA using data from cache
	 */
	protected void tryContinueLBKeoghRQ(double scoreToBeat) {
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
	 * The main function for LazyUCR. 
	 * Start with LBKim,LBKeogh(Q,R),LBKeogh(R,Q),DTW 
	 * @param scoreToBeat
	 * @param w
	 * @return
	 */
	public RefineReturnType tryToBeat(double scoreToBeat, int w) {
		setCurrentW(w);
		
		switch (status) {
		case Previous_Window_LB:
		case Previous_Window_DTW:
		case LB_Kim:
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			// if LB_Kim_FL done, then start LB_Keogh(Q,R)
			indexStoppedLB = 0;
			minDist = 0;
		case Partial_LB_KeoghQR:
			// if had started LB_Keogh, then just starting from
			// previous index
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			tryContinueLBKeoghQR(scoreToBeat);
			if(minDist>bestMinDist){
				bestMinDist = minDist;
			}
			if (bestMinDist >= scoreToBeat) {
				// stopped in the middle so must be pruning
				if (indexStoppedLB < query.getNbTuples()) {
					status = LBStatus.Partial_LB_KeoghQR;
				} else {
					LBKeogh1 = minDist;
					status = LBStatus.Full_LB_KeoghQR;
				}
				return RefineReturnType.Pruned_with_LB;
			}else{
				status = LBStatus.Full_LB_KeoghQR;
			}
		case Full_LB_KeoghQR:
			// if LB_Keogh(Q,R) has been done, then we do the second one
			indexStoppedLB = 0;
			minDist = 0;
		case Partial_LB_KeoghRQ:
			// if had started LB_Keogh, then just starting from
			// previous index
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			tryContinueLBKeoghRQ(scoreToBeat);
			if(minDist>bestMinDist){
				bestMinDist = minDist;
			}
			if (bestMinDist >= scoreToBeat) {
				if (indexStoppedLB < reference.getNbTuples()) {
					status = LBStatus.Partial_LB_KeoghRQ;
				} else {
					LBKeogh2 = minDist;
					status = LBStatus.Full_LB_KeoghRQ;
				}
				return RefineReturnType.Pruned_with_LB;
			}else{
				status = LBStatus.Full_LB_KeoghRQ;
			}
		case Full_LB_KeoghRQ:
			// if had finished LB_Keogh(R,Q), then DTW
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			double res = query.DTW(reference, currentW);
			minDist = res * res;
			if(minDist>bestMinDist){
				bestMinDist = minDist;
			}
			status = LBStatus.Full_DTW;
		case Full_DTW:
			if (bestMinDist >= scoreToBeat) {
				return RefineReturnType.Pruned_with_DTW;
			} else {
				return RefineReturnType.New_best;
			}
		default:
			throw new RuntimeException("Case not managed");
		}
	}
	
	/**
	 * The main function for LazyUCR with PrunedDTW. 
	 * Start with LBKim,LBKeogh(Q,R),LBKeogh(R,Q),PrunedDTW 
	 * @param scoreToBeat
	 * @param w
	 * @return
	 */
	public RefineReturnType tryToBeatPrunedDTW(double scoreToBeat, int w) {
		setCurrentW(w);
		
		switch (status) {
		case Previous_Window_LB:
		case Previous_Window_DTW:
		case LB_Kim:
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			// if LB_Kim_FL done, then start LB_Keogh(Q,R)
			indexStoppedLB = 0;
			minDist = 0;
		case Partial_LB_KeoghQR:
			// if had started LB_Keogh, then just starting from
			// previous index
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			tryContinueLBKeoghQR(scoreToBeat);
			if(minDist>bestMinDist){
				bestMinDist = minDist;
			}
			if (bestMinDist >= scoreToBeat) {
				// stopped in the middle so must be pruning
				if (indexStoppedLB < query.getNbTuples()) {
					status = LBStatus.Partial_LB_KeoghQR;
				} else {
					LBKeogh1 = minDist;
					status = LBStatus.Full_LB_KeoghQR;
				}
				return RefineReturnType.Pruned_with_LB;
			}else{
				status = LBStatus.Full_LB_KeoghQR;
			}
		case Full_LB_KeoghQR:
			// if LB_Keogh(Q,R) has been done, then we do the second one
			indexStoppedLB = 0;
			minDist = 0;
		case Partial_LB_KeoghRQ:
			// if had started LB_Keogh, then just starting from
			// previous index
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			tryContinueLBKeoghRQ(scoreToBeat);
			if(minDist>bestMinDist){
				bestMinDist = minDist;
			}
			if (bestMinDist >= scoreToBeat) {
				if (indexStoppedLB < reference.getNbTuples()) {
					status = LBStatus.Partial_LB_KeoghRQ;
				} else {
					LBKeogh2 = minDist;
					status = LBStatus.Full_LB_KeoghRQ;
				}
				return RefineReturnType.Pruned_with_LB;
			}else{
				status = LBStatus.Full_LB_KeoghRQ;
			}
		case Full_LB_KeoghRQ:
			// if had finished LB_Keogh(R,Q), then DTW
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			double res = query.PrunedDTW(reference, currentW);
			minDist = res * res;
			if(minDist>bestMinDist){
				bestMinDist = minDist;
			}
			status = LBStatus.Full_DTW;
		case Full_DTW:
			if (bestMinDist >= scoreToBeat) {
				return RefineReturnType.Pruned_with_DTW;
			} else {
				return RefineReturnType.New_best;
			}
		default:
			throw new RuntimeException("Case not managed");
		}
	}
	
	/**
	 * The main function for LazyUCR with PrunedDTW with an Upper Bound 
	 * Start with LBKim,LBKeogh(Q,R),LBKeogh(R,Q),PrunedDTW 
	 * @param scoreToBeat
	 * @param w
	 * @return
	 */
	public RefineReturnType tryToBeatPrunedDTW(double scoreToBeat, int w, double UB) {
		setCurrentW(w);
		
		switch (status) {
		case Previous_Window_LB:
		case Previous_Window_DTW:
		case LB_Kim:
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			// if LB_Kim_FL done, then start LB_Keogh(Q,R)
			indexStoppedLB = 0;
			minDist = 0;
		case Partial_LB_KeoghQR:
			// if had started LB_Keogh, then just starting from
			// previous index
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			tryContinueLBKeoghQR(scoreToBeat);
			if(minDist>bestMinDist){
				bestMinDist = minDist;
			}
			if (bestMinDist >= scoreToBeat) {
				// stopped in the middle so must be pruning
				if (indexStoppedLB < query.getNbTuples()) {
					status = LBStatus.Partial_LB_KeoghQR;
				} else {
					LBKeogh1 = minDist;
					status = LBStatus.Full_LB_KeoghQR;
				}
				return RefineReturnType.Pruned_with_LB;
			}else{
				status = LBStatus.Full_LB_KeoghQR;
			}
		case Full_LB_KeoghQR:
			// if LB_Keogh(Q,R) has been done, then we do the second one
			indexStoppedLB = 0;
			minDist = 0;
		case Partial_LB_KeoghRQ:
			// if had started LB_Keogh, then just starting from
			// previous index
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			tryContinueLBKeoghRQ(scoreToBeat);
			if(minDist>bestMinDist){
				bestMinDist = minDist;
			}
			if (bestMinDist >= scoreToBeat) {
				if (indexStoppedLB < reference.getNbTuples()) {
					status = LBStatus.Partial_LB_KeoghRQ;
				} else {
					LBKeogh2 = minDist;
					status = LBStatus.Full_LB_KeoghRQ;
				}
				return RefineReturnType.Pruned_with_LB;
			}else{
				status = LBStatus.Full_LB_KeoghRQ;
			}
		case Full_LB_KeoghRQ:
			// if had finished LB_Keogh(R,Q), then PrunedDTW
			if(bestMinDist>=scoreToBeat){
				return RefineReturnType.Pruned_with_LB;
			}
			double res = query.PrunedDTW(reference, currentW, UB);
			minDist = res * res;
			if(minDist>bestMinDist){
				bestMinDist = minDist;
			}
			status = LBStatus.Full_DTW;
		case Full_DTW:
			if (bestMinDist >= scoreToBeat) {
				return RefineReturnType.Pruned_with_DTW;
			} else {
				return RefineReturnType.New_best;
			}
		default:
			throw new RuntimeException("Case not managed");
		}
	}

	@Override
	public String toString() {
		return "" + indexQuery+ " - "+indexReference+" - "+bestMinDist;
	}

	public int getOtherIndex(int index) {
		if (index == indexQuery) {
			return indexReference;
		} else {
			return indexQuery;
		}
	}

	public SymbolicSequence getSequenceForOtherIndex(int index) {
		if (index == indexQuery) {
			return reference;
		} else {
			return query;
		}
	}

	public double getDistance(int window) {
		// System.out.println(minDist+" - "+minWindowValidityFullDTW + "
		// - "+window+ " - "+status.name());
		if (status == LBStatus.Full_DTW && minWindowValidityFullDTW <= window) {
			return minDist;
		}
		throw new RuntimeException("Shouldn't call getDistance if not sure there is a valid already-computed DTW distance");
	}

	public int getMinWindowValidityForFullDistance() {
		if (status == LBStatus.Full_DTW) {
			return minWindowValidityFullDTW;
		}
		throw new RuntimeException("Shouldn't call getDistance if not sure there is a valid already-computed DTW distance");
	}

	public double[] getUBPartial() {
		return ubPartials;
	}
	
	public double getEuclideanDistance() {
		return EuclideanDist;
	}
	
	@Override
	public int compareTo(LazyAssessNNEarlyAbandon o) {
		int res = this.compare(o);
		return res;
		
	}
	
	protected int compare(LazyAssessNNEarlyAbandon o) {
		double num1 = this.getDoubleValueForRanking();
		double num2 = o.getDoubleValueForRanking();
		return Double.compare(num1, num2);
	}
	
	protected double getDoubleValueForRanking() {
		double thisD = this.bestMinDist;
		
		switch(status){
		case Full_DTW:
		case Full_LB_KeoghQR:
		case Full_LB_KeoghRQ:
			return thisD/query.getNbTuples();
		case LB_Kim:
			return thisD/nOperationsLBKim;
		case Partial_LB_KeoghQR:
		case Partial_LB_KeoghRQ:
			return thisD/indexStoppedLB;
		case Previous_Window_DTW:
			return 0.8*thisD/query.getNbTuples();	// DTW(w+1) should be tighter
		case Previous_Window_LB:
			if(indexStoppedLB==0){
				//lb kim
				return thisD/nOperationsLBKim;
			}else{
				//lbkeogh
				return thisD/oldIndexStoppedLB;
			}
		default: 
			throw new RuntimeException("shouldn't come here");
		}

	}
	
	@Override
	public boolean equals(Object o) {
		LazyAssessNNEarlyAbandon d = (LazyAssessNNEarlyAbandon) o;
		return (this.indexQuery == d.indexQuery && this.indexReference == d.indexReference);
	}

	public LBStatus getStatus() {
		return status;
	}
	
	public void setFullDistStatus(){
		this.status = LBStatus.Full_DTW;
	}
	
	public double getBestLB(){
		return bestMinDist;
	}
	
}
