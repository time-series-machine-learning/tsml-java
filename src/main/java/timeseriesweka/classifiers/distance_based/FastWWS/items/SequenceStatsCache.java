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

import java.util.Arrays;

import timeseriesweka.classifiers.distance_based.FastWWS.sequences.SymbolicSequence;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * Cache for storing the information on the time series dataset
 * 
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 *
 */
public class SequenceStatsCache {
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Fields
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	double[][]LEs,UEs;
	double []mins,maxs;
	boolean[]isMinFirst,isMinLast,isMaxFirst,isMaxLast;
	int[]lastWindowComputed;
	int currentWindow;
	SymbolicSequence[]train;
	IndexedDouble[][]indicesSortedByAbsoluteValue;
	
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	/**
	 * Initialisation and find min,max for every time series in the dataset
	 * @param train
	 * @param startingWindow
	 */
	public SequenceStatsCache(SymbolicSequence[]train,int startingWindow){
		this.train = train;
		int nSequences = train.length;
		int length = train[0].getNbTuples();
		this.LEs = new double[nSequences][length];
		this.UEs = new double[nSequences][length];
		this.lastWindowComputed = new int[nSequences];
		Arrays.fill(this.lastWindowComputed, -1);
		this.currentWindow = startingWindow;
		this.mins = new double[nSequences];
		this.maxs = new double[nSequences];
		this.isMinFirst = new boolean[nSequences];
		this.isMinLast = new boolean[nSequences];
		this.isMaxFirst = new boolean[nSequences];
		this.isMaxLast = new boolean[nSequences];
		this.indicesSortedByAbsoluteValue = new IndexedDouble[nSequences][length];
		for(int i=0;i<train.length;i++){
			double min=Double.POSITIVE_INFINITY;
			double max = Double.NEGATIVE_INFINITY;
			int indexMin=-1,indexMax=-1;
			for(int j=0;j<train[i].getNbTuples();j++){
				double elt = ((MonoDoubleItemSet)train[i].sequence[j]).value;
				if(elt>max){
					max = elt;
					indexMax = j;
				}
				if(elt<min){
					min = elt;
					indexMin = j;
				}
				indicesSortedByAbsoluteValue[i][j]=new IndexedDouble(j, Math.abs(elt));
			}
			mins[i]=min;
			maxs[i]=max;
			isMinFirst[i]=(indexMin==0);
			isMinLast[i]=(indexMin==(train[i].getNbTuples()-1));
			isMaxFirst[i]=(indexMax==0);
			isMaxLast[i]=(indexMax==(train[i].getNbTuples()-1));
			Arrays.sort(indicesSortedByAbsoluteValue[i], (v1,v2)-> -Double.compare(v1.value, v2.value));
		}
	}
	
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Methods
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	/**
	 * Get lower envelope of the ith time series,
	 * Compute it if not computed
	 * @param i
	 * @param w
	 * @return
	 */
	public double[] getLE(int i,int w){
		if(lastWindowComputed[i]!=w){
			computeLEandUE(i,w);
		}
		return LEs[i];
	}
	
	/**
	 * Get upper envelope of the ith time series,
	 * Compute it if not computed
	 * @param i
	 * @param w
	 * @return
	 */
	public double[] getUE(int i,int w){
		if(lastWindowComputed[i]!=w){
			computeLEandUE(i,w);
		}
		return UEs[i];
	}
	
	/**
	 * Compute envelope for the ith time series with window w
	 * @param i
	 * @param w
	 */
	protected void computeLEandUE(int i,int w){
		train[i].LB_KeoghFillUL(w, UEs[i], LEs[i]);
		this.lastWindowComputed[i]=w;
	}
	
	public boolean isMinFirst(int i){
		return isMinFirst[i];
	}
	
	public boolean isMaxFirst(int i){
		return isMaxFirst[i];
	}
	public boolean isMinLast(int i){
		return isMinLast[i];
	}
	
	public boolean isMaxLast(int i){
		return isMaxLast[i];
	}
	
	public double getMin(int i){
		return mins[i];
	}
	
	public double getMax(int i){
		return maxs[i];
	}	
	
	class IndexedDouble{
		double value;
		int index;
		public IndexedDouble(int index,double value){
			this.value = value;
			this.index = index;
		}
	}
	
	public int getIndexNthHighestVal(int i,int n){
		return indicesSortedByAbsoluteValue[i][n].index;
	}
}
