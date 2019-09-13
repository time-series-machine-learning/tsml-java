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
package timeseriesweka.classifiers.distance_based.FastWWS.sequences;

import static java.lang.Math.sqrt;

import java.util.ArrayList;
import java.util.Arrays;

import timeseriesweka.classifiers.distance_based.FastWWS.items.DTWResult;
import timeseriesweka.classifiers.distance_based.FastWWS.items.Itemset;
import timeseriesweka.classifiers.distance_based.FastWWS.items.MonoDoubleItemSet;
import timeseriesweka.classifiers.distance_based.FastWWS.items.MonoItemSet;
import timeseriesweka.classifiers.distance_based.FastWWS.tools.Tools;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * Class for time series
 * 
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 *
 */
public class SymbolicSequence implements java.io.Serializable {
	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Internal types
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	private static final long serialVersionUID = -8340081464719919763L;

	public static int w = Integer.MAX_VALUE;

	protected final static int NB_ITERATIONS = 15;

	protected final static int RIEN = -1;
	protected final static int DIAGONALE = 0;
	protected final static int GAUCHE = 1;
	protected final static int HAUT = 2;

	public Itemset[] sequence;

	public static long nDTWExt;

	private final static int MAX_SEQ_LENGTH = 8000;

	public static double[][] matriceW = new double[SymbolicSequence.MAX_SEQ_LENGTH][SymbolicSequence.MAX_SEQ_LENGTH];
	public static int[][] matriceChoix = new int[SymbolicSequence.MAX_SEQ_LENGTH][SymbolicSequence.MAX_SEQ_LENGTH];
	protected static int[][] optimalPathLength = new int[SymbolicSequence.MAX_SEQ_LENGTH][SymbolicSequence.MAX_SEQ_LENGTH];
	protected static int[][] minWarpingWindow = new int[SymbolicSequence.MAX_SEQ_LENGTH][SymbolicSequence.MAX_SEQ_LENGTH];
	public static double[] ub_partials = new double[SymbolicSequence.MAX_SEQ_LENGTH];

	public SymbolicSequence(final Itemset[] sequence) {
		if (sequence == null || sequence.length == 0) {
			throw new RuntimeException("sequence vide");
		}
		this.sequence = sequence;
	}

	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constructor
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	public SymbolicSequence(SymbolicSequence o) {
		if (o.sequence == null || o.sequence.length == 0) {
			throw new RuntimeException("sequence vide");
		}
		this.sequence = o.sequence;
	}

	// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Methods
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
	@Override
	public Object clone() {
		final Itemset[] newSequence = Arrays.copyOf(sequence, sequence.length);
		for (int i = 0; i < newSequence.length; i++) {
			newSequence[i] = sequence[i].clone();
		}

		return new SymbolicSequence(newSequence);
	}

	/**
	 * Get data point from this sequence at index n
	 * @param n
	 * @return
	 */
	public Itemset getItem(final int n) {
		return sequence[n];
	}

	/**
	 * Return the length of this sequence
	 * @return
	 */
	public final int getNbTuples() {
		return this.sequence.length;
	}

	/** 
	 * Euclidean distance to a
	 * @param a
	 * @return
	 */
	public final double distanceEuc(SymbolicSequence a) {
		final int length = this.getNbTuples();

		double res = 0;
		for (int i = 0; i < length; i++) {
			res += this.sequence[i].squaredDistance(a.sequence[i]);
		}
		return sqrt(res);
	}
	
	/**
	 * Squared Euclidean distance with early abandon
	 * @param a
	 * @param max
	 * @return
	 */
	public final double squaredEucEarlyAbandon(SymbolicSequence a,double max) {
		Itemset[] series1 = this.sequence;
		Itemset[] series2 = a.sequence;
		int minLength = Math.min(series1.length, series2.length);
		double distance = 0;
		for (int i = 0; i < minLength; i++) {
			distance += series1[i].squaredDistance(series2[i]);
			if(distance>= max){
				return Double.MAX_VALUE;
			}
		}
		return distance;
	}
	
	/**
	 * Euclidean distance with normalised length
	 * @param a
	 * @return
	 */
	public final double distanceEucNormalized(SymbolicSequence a) {
		Itemset[] series1 = this.sequence;
		Itemset[] series2 = a.sequence;
		int minLength = Math.min(series1.length, series2.length);
		double distance = 0;
		for (int i = 0; i < minLength; i++) {
			distance += series1[i].squaredDistance(series2[i]);
		}
		return sqrt(distance)/minLength;
	}
	
	/**
	 * Squared Euclidean distance with early abandon and normalised length
	 * @param a
	 * @param max
	 * @return
	 */
	public final double squaredEucEarlyAbandonNormalized(SymbolicSequence a,double max) {
		Itemset[] series1 = this.sequence;
		Itemset[] series2 = a.sequence;
		int minLength = Math.min(series1.length, series2.length);
		double distance = 0;
		for (int i = 0; i < minLength; i++) {
			distance += series1[i].squaredDistance(series2[i]);
			if(distance/(i+1)>= max){
				return Double.MAX_VALUE;
			}
		}
		return distance/minLength;
	}

	/**
	 * Compute LB Keogh to a at window r
	 * @param a
	 * @param r
	 * @return
	 */
	public final double LB_Keogh(SymbolicSequence a, int r) {
		final int length = Math.min(this.getNbTuples(), a.getNbTuples());

		double[] U = new double[length];
		double[] L = new double[length];

		for (int i = 0; i < length; i++) {
			double min = Double.POSITIVE_INFINITY;
			double max = Double.NEGATIVE_INFINITY;
			int startR = Math.max(0, i - r);
			int stopR = Math.min(length - 1, i + r);
			for (int j = startR; j <= stopR; j++) {
				double value = ((MonoDoubleItemSet) this.sequence[j]).value;
				min = Math.min(min, value);
				max = Math.max(max, value);
			}
			L[i] = min;
			U[i] = max;
		}

		double res = 0;
		for (int i = 0; i < length; i++) {
			double c = ((MonoDoubleItemSet) a.sequence[i]).value;
			if (c < L[i]) {
				double diff = L[i] - c;
				res += diff * diff;
			} else if (U[i] < c) {
				double diff = U[i] - c;
				res += diff * diff;
			}
		}

		return sqrt(res);
	}
	
	/**
	 * Compute LB Keogh with a given U and L envelope
	 * @param a
	 * @param r
	 * @param U
	 * @param L
	 * @return
	 */
	public final double LB_Keogh(SymbolicSequence a, int r,double[]U,double[]L) {
		LB_KeoghFillUL(r,U,L);
		return LB_KeoghPreFilled(a, U, L);
	}
	
	/**
	 * Fill the Upper and Lower Envelope for this sequence
	 * @param r
	 * @param U
	 * @param L
	 */
	public final void LB_KeoghFillUL( int r,double[]U,double[]L) {
		final int length = this.getNbTuples();

		for (int i = 0; i < length; i++) {
			double min = Double.POSITIVE_INFINITY;
			double max = Double.NEGATIVE_INFINITY;
			int startR = Math.max(0, i - r);
			int stopR = Math.min(length - 1, i + r);
			for (int j = startR; j <= stopR; j++) {
				double value = ((MonoDoubleItemSet) this.sequence[j]).value;
				min = Math.min(min, value);
				max = Math.max(max, value);
			}
			L[i] = min;
			U[i] = max;
		}
	}
	
	/** 
	 * Compute LB Keogh with a prefilled U and L Envelope
	 * @param a
	 * @param U
	 * @param L
	 * @return
	 */
	public static final double LB_KeoghPreFilled(SymbolicSequence a, double[]U,double[]L) {
		final int length = Math.min(U.length, a.getNbTuples());

		double res = 0;
		for (int i = 0; i < length; i++) {
			double c = ((MonoDoubleItemSet) a.sequence[i]).value;
			if (c < L[i]) {
				double diff = L[i] - c;
				res += diff * diff;
			} else if (U[i] < c) {
				double diff = U[i] - c;
				res += diff * diff;
			}
		}

		return sqrt(res);
	}
	
	/**
	 * Compute Full DTW distance to sequence a without window validity
	 * @param a
	 * @return
	 */
	public synchronized double distance(SymbolicSequence a) {
		return this.distance(a, matriceW);
	}

	/** 
	 * Compute Full DTW distance with cost matrix without window validity
	 * @param a
	 * @param matriceW
	 * @return
	 */
	public double distance(SymbolicSequence a, double[][] matriceW) {
		SymbolicSequence S1 = this;
		SymbolicSequence S2 = a;

		final int tailleS = S1.getNbTuples();
		final int tailleT = S2.getNbTuples();

		int i, j;
		matriceW[0][0] = S1.sequence[0].squaredDistance(S2.sequence[0]);
		for (i = 1; i < tailleS; i++) {
			matriceW[i][0] = matriceW[i - 1][0]
					+ S1.sequence[i].squaredDistance(S2.sequence[0]);
		}

		for (j = 1; j < tailleT; j++) {
			matriceW[0][j] = matriceW[0][j - 1]
					+ S1.sequence[0].squaredDistance(S2.sequence[j]);
		}

		for (i = 1; i < tailleS; i++) {
			for (j = 1; j < tailleT; j++) {
				matriceW[i][j] = Tools.Min3(matriceW[i - 1][j - 1],
						matriceW[i][j - 1], matriceW[i - 1][j])
						+ S1.sequence[i].squaredDistance(S2.sequence[j]);
			}
		}
		
		return sqrt(matriceW[tailleS - 1][tailleT - 1]);
	}
	
	/** 
	 * Compute Euclidean Distance
	 * @param a
	 * @return
	 */
	public double ED(SymbolicSequence a) {
		double sum = 0;
		for (int i = 0; i < this.getNbTuples(); i++) {
			sum += this.sequence[i].squaredDistance(a.sequence[i]);
		}

		return sqrt(sum);
	}
	
	/**
	 * Compute DTW distance with warping window without window validity
	 * @param a
	 * @param w
	 * @return
	 */
	public synchronized double DTW(SymbolicSequence a,int w) {
		return this.DTW(a, w, matriceW);
	}
	
	/** 
	 * Compute DTW distance with warping window and cost matrix without window validity
	 * @param a
	 * @param w
	 * @param warpingMatrix
	 * @return
	 */
	public double DTW(SymbolicSequence a, int w,double[][] warpingMatrix) {
		final int length1 = this.getNbTuples();
		final int length2 = a.getNbTuples();

		int i, j;
		warpingMatrix[0][0] = this.sequence[0].squaredDistance(a.sequence[0]);
		for (i = 1; i < Math.min(length1, 1 + w); i++) {
			warpingMatrix[i][0] = warpingMatrix[i - 1][0] + this.sequence[i].squaredDistance(a.sequence[0]);
		}

		for (j = 1; j < Math.min(length2, 1 + w); j++) {
			warpingMatrix[0][j] = warpingMatrix[0][j - 1] + this.sequence[0].squaredDistance(a.sequence[j]);
		}
		if (j < length2) {
			warpingMatrix[0][j] = Double.POSITIVE_INFINITY;
		}

		for (i = 1; i < length1; i++) {
			int jStart = Math.max(1, i - w);
			int jStop = Math.min(length2, i + w + 1);
			int indexInftyLeft = i-w-1;
			if(indexInftyLeft>=0)warpingMatrix[i][indexInftyLeft] = Double.POSITIVE_INFINITY;

			for (j = jStart; j < jStop; j++) {
				warpingMatrix[i][j] = Tools.Min3(warpingMatrix[i - 1][j - 1],
						warpingMatrix[i][j - 1], warpingMatrix[i - 1][j])
						+ this.sequence[i].squaredDistance(a.sequence[j]);
			}
			if (jStop < length2) {
				warpingMatrix[i][jStop] = Double.POSITIVE_INFINITY;
			}
		}

		return sqrt(warpingMatrix[length1 - 1][length2 - 1]);
	}
	
	/** 
	 * Compute PrunedDTW with warping window using partial Upper bound 
	 * that is computed on the go without window validity
	 * @param T
	 * @param w
	 * @return
	 */
	public double PrunedDTW(SymbolicSequence T, int w) {
		nDTWExt++;
		final int tailleS = this.getNbTuples();
		final int tailleT = T.getNbTuples();
		int i, j, indiceRes;
		double res = 0.0;

		int sc = 1, ec = 1, ec_next = 0;
		boolean found_lower;
		double UB = 0;

		ub_partials[tailleS] = 0;
		for (i = tailleS-1; i >= 0; i--) {
			ub_partials[i] = ub_partials[i+1] + this.sequence[i].squaredDistance(T.sequence[i]);
			matriceW[i][0] = Double.POSITIVE_INFINITY;
			matriceW[0][i] = Double.POSITIVE_INFINITY;
		}
		matriceW[tailleS][0] = Double.POSITIVE_INFINITY;
		matriceW[0][tailleS] = Double.POSITIVE_INFINITY;
		matriceW[0][0] = 0;
		
		UB = ub_partials[0];
		
		for (i = 1; i <= tailleS; i++) {
			int jStart = Math.max(sc,  i-w);
			int jStop = Math.min(i+w, tailleT);
			UB = ub_partials[i-1] + matriceW[i-1][i-1];
			matriceW[i][jStart-1] = Double.POSITIVE_INFINITY;
			
			found_lower = false;
			for (j = jStart; j <= jStop; j++) {
				if (j > ec) {
					res = matriceW[i][j-1];
				} else {
					indiceRes = Tools.ArgMin3(matriceW[i - 1][j - 1],
							matriceW[i][j - 1], matriceW[i - 1][j]);
					switch (indiceRes) {
					case DIAGONALE:
						res = matriceW[i - 1][j - 1];
						break;
					case GAUCHE:
						res = matriceW[i][j - 1];
						break;
					case HAUT: 
						res = matriceW[i - 1][j];
						break;
					}
				}
				matriceW[i][j] = this.sequence[i-1].squaredDistance(T.sequence[j-1]) + res;
				
				if (matriceW[i][j] > UB) {
					if (!found_lower) {
						sc = j+1;
					}
					if (j > ec) {
						matriceW[i][j+1] = Double.POSITIVE_INFINITY;
						break;
					}
				} else {
					found_lower = true;
					ec_next = j;
				}
				
				if (jStop + 1 <= tailleT) {
					matriceW[i][jStop+1] = Double.POSITIVE_INFINITY;
				}
			}
			ec_next++;
			ec = ec_next;
		}
		
		return sqrt(matriceW[tailleS][tailleT]);
	}
	
	/**
	 * Compute PrunedDTW with warping window using a given upper bound without window validity
	 * @param T
	 * @param w
	 * @param UB
	 * @return
	 */
	public double PrunedDTW(SymbolicSequence T, int w, double UB) {
		nDTWExt++;
		final int tailleS = this.getNbTuples();
		final int tailleT = T.getNbTuples();
		int i, j, indiceRes;
		double res = 0.0;

		int sc = 1, ec = 1, ec_next = 0;
		boolean found_lower;

		for (i = tailleS-1; i >= 0; i--) {
			matriceW[i][0] = Double.POSITIVE_INFINITY;
			matriceW[0][i] = Double.POSITIVE_INFINITY;
		}
		matriceW[tailleS][0] = Double.POSITIVE_INFINITY;
		matriceW[0][tailleS] = Double.POSITIVE_INFINITY;
		matriceW[0][0] = 0;
		
		for (i = 1; i <= tailleS; i++) {
			int jStart = Math.max(sc,  i-w);
			int jStop = Math.min(i+w, tailleT);
			matriceW[i][jStart-1] = Double.POSITIVE_INFINITY;
			
			found_lower = false;
			for (j = jStart; j <= jStop; j++) {
				if (j > ec) {
					res = matriceW[i][j-1];
				} else {
					indiceRes = Tools.ArgMin3(matriceW[i - 1][j - 1],
							matriceW[i][j - 1], matriceW[i - 1][j]);
					switch (indiceRes) {
					case DIAGONALE:
						res = matriceW[i - 1][j - 1];
						break;
					case GAUCHE:
						res = matriceW[i][j - 1];
						break;
					case HAUT: 
						res = matriceW[i - 1][j];
						break;
					}
				}
				matriceW[i][j] = this.sequence[i-1].squaredDistance(T.sequence[j-1]) + res;
				
				if (matriceW[i][j] > UB) {
					if (!found_lower) {
						sc = j+1;
					}
					if (j > ec) {
						matriceW[i][j+1] = Double.POSITIVE_INFINITY;
						break;
					}
				} else {
					found_lower = true;
					ec_next = j;
				}
				
				if (jStop + 1 <= tailleT) {
					matriceW[i][jStop+1] = Double.POSITIVE_INFINITY;
				}
				
			}
			ec_next++;
			ec = ec_next;
		}
		
		return sqrt(matriceW[tailleS][tailleT]);
	}
	
	/**
	 * Compute DTW with warping window and window validity
	 * @param T
	 * @param w
	 * @return
	 */
	public synchronized DTWResult DTWExtResults(SymbolicSequence T, int w) {
		nDTWExt++;
		final int tailleS = this.getNbTuples();
		final int tailleT = T.getNbTuples();
		int i, j, indiceRes;
		double res = 0.0;
		
		matriceW[0][0] = this.sequence[0].squaredDistance(T.sequence[0]);
		minWarpingWindow[0][0]=0;
		for (i = 1; i < Math.min(tailleS, 1 + w); i++) {
			matriceW[i][0] = matriceW[i - 1][0]+ this.sequence[i].squaredDistance(T.sequence[0]);
			minWarpingWindow[i][0]=i;
		}
		
		for (j = 1; j < Math.min(tailleT, 1 + w); j++) {
			matriceW[0][j] = matriceW[0][j - 1]+ T.sequence[j].squaredDistance(sequence[0]);
			minWarpingWindow[0][j] = j;
		}
		if (j < tailleT) {
			matriceW[0][j] = Double.POSITIVE_INFINITY;
		}

		for (i = 1; i < tailleS; i++) {
			int jStart = Math.max(1, i - w);
			int jStop = Math.min(tailleT, i + w + 1);
			int indexInftyLeft = i-w-1;
			if(indexInftyLeft>=0)
				matriceW[i][indexInftyLeft] = Double.POSITIVE_INFINITY;
			for (j = jStart; j < jStop; j++) {
				indiceRes = Tools.ArgMin3(matriceW[i - 1][j - 1],
								matriceW[i][j - 1], matriceW[i - 1][j]);
				int absIJ = Math.abs(i-j);
				switch (indiceRes) {
				case DIAGONALE:
					res = matriceW[i - 1][j - 1];
					minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i-1][j-1]);
					break;
				case GAUCHE:
					res = matriceW[i][j - 1];
					minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i][j-1]);
					break;
				case HAUT:
					res = matriceW[i - 1][j];
					minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i-1][j]);
					break;
				}
				matriceW[i][j] = res
						+ this.sequence[i].squaredDistance(T.sequence[j]);
			}
			if (j < tailleT) {
				matriceW[i][j] = Double.POSITIVE_INFINITY;
			}
		}
		
		DTWResult resExt= new DTWResult();
		resExt.distance = sqrt(matriceW[tailleS - 1][tailleT- 1]);
		resExt.r = minWarpingWindow[tailleS - 1][tailleT- 1];
		return resExt;
	}
		
	/** 
	 * Compute PrunedDTW with warping window and window validity
	 * @param T
	 * @param w
	 * @return
	 */
	public synchronized DTWResult PrunedDTWExtResults(SymbolicSequence T, int w) {
		nDTWExt++;
		final int tailleS = this.getNbTuples();
		final int tailleT = T.getNbTuples();
		int i, j, indiceRes;
		double res = 0.0;

		int sc = 1, ec = 1, ec_next = 0;
		boolean found_lower;
		double UB = 0;
		ub_partials[tailleS] = 0;
		for (i = tailleS-1; i >= 0; i--) {
			ub_partials[i] = ub_partials[i+1] + this.sequence[i].squaredDistance(T.sequence[i]);
			matriceW[i][0] = Double.POSITIVE_INFINITY;
			matriceW[0][i] = Double.POSITIVE_INFINITY;
			minWarpingWindow[0][i] = i;
			minWarpingWindow[i][0] = i;
		}
		matriceW[tailleS][0] = Double.POSITIVE_INFINITY;
		matriceW[0][tailleS] = Double.POSITIVE_INFINITY;
		matriceW[0][0] = 0;
		
		UB = ub_partials[0];
		
		for (i = 1; i <= tailleS; i++) {
			int jStart = Math.max(sc,  i-w);
			int jStop = Math.min(i+w, tailleT);
			UB = ub_partials[i-1] + matriceW[i-1][i-1];
			matriceW[i][jStart-1] = Double.POSITIVE_INFINITY;
			
			found_lower = false;
			for (j = jStart; j <= jStop; j++) {
				int absIJ = Math.abs((i-1)-(j-1));
				if (j > ec) {
					res = matriceW[i][j-1];
					if (i == 1)
						minWarpingWindow[i][j] = i;
					else
						minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i][j-1]);
				} else {
					indiceRes = Tools.ArgMin3(matriceW[i - 1][j - 1],
							matriceW[i][j - 1], matriceW[i - 1][j]);
					switch (indiceRes) {
					case DIAGONALE:
						res = matriceW[i - 1][j - 1];
						minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i-1][j-1]);
						break;
					case GAUCHE:
						res = matriceW[i][j - 1];
						if (i == 1)
							minWarpingWindow[i][j] = i;
						else
							minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i][j-1]);
						break;
					case HAUT: 
						res = matriceW[i - 1][j];
						if (j == 1) 
							minWarpingWindow[i][j] = j;
						else
							minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i-1][j]);
						break;
					}
				}
				matriceW[i][j] = this.sequence[i-1].squaredDistance(T.sequence[j-1]) + res;
				
				if (jStop + 1 <= tailleT) {
					matriceW[i][jStop+1] = Double.POSITIVE_INFINITY;
				}
				if (matriceW[i][j] > UB) {
					if (!found_lower) {
						sc = j+1;
					}
					if (j > ec) {
						matriceW[i][j+1] = Double.POSITIVE_INFINITY;
						break;
					}
				} else {
					found_lower = true;
					ec_next = j;
				}
			}
			ec_next++;
			ec = ec_next;
		}
				
		DTWResult resExt= new DTWResult();
		resExt.distance = sqrt(matriceW[tailleS][tailleT]);
		resExt.r = minWarpingWindow[tailleS][tailleT];
		return resExt;
	}

	public synchronized ArrayList<Integer>[] DTWAssociationFromS(
			final SymbolicSequence T) {

		@SuppressWarnings("unchecked")
		final ArrayList<Integer>[] association = new ArrayList[this.getNbTuples()];
		for (int i = 0; i < association.length; i++) {
			association[i] = new ArrayList<Integer>();
		}
		final int tailleS = this.getNbTuples();
		final int tailleT = T.getNbTuples();
		int nbTuplesAverageSeq, i, j, indiceRes;
		double res = 0.0;

		matriceW[0][0] = this.sequence[0].squaredDistance(T.sequence[0]);
		matriceChoix[0][0] = RIEN;
		optimalPathLength[0][0] = 0;

		for (i = 1; i < tailleS; i++) {
			matriceW[i][0] = matriceW[i - 1][0]
					+ this.sequence[i].squaredDistance(T.sequence[0]);
			matriceChoix[i][0] = HAUT;
			optimalPathLength[i][0] = i;
		}
		for (j = 1; j < tailleT; j++) {
			matriceW[0][j] = matriceW[0][j - 1]
					+ T.sequence[j].squaredDistance(sequence[0]);
			matriceChoix[0][j] = GAUCHE;
			optimalPathLength[0][j] = j;
		}

		for (i = 1; i < tailleS; i++) {
			for (j = 1; j < tailleT; j++) {
				indiceRes = Tools.ArgMin3(matriceW[i - 1][j - 1],
						matriceW[i][j - 1], matriceW[i - 1][j]);
				matriceChoix[i][j] = indiceRes;
				switch (indiceRes) {
				case DIAGONALE:
					res = matriceW[i - 1][j - 1];
					optimalPathLength[i][j] = optimalPathLength[i - 1][j - 1] + 1;
					break;
				case GAUCHE:
					res = matriceW[i][j - 1];
					optimalPathLength[i][j] = optimalPathLength[i][j - 1] + 1;
					break;
				case HAUT:
					res = matriceW[i - 1][j];
					optimalPathLength[i][j] = optimalPathLength[i - 1][j] + 1;
					break;
				}
				matriceW[i][j] = res
						+ this.sequence[i].squaredDistance(T.sequence[j]);
			}
		}

		nbTuplesAverageSeq = optimalPathLength[tailleS - 1][tailleT - 1] + 1;

		i = tailleS - 1;
		j = tailleT - 1;

		for (int t = nbTuplesAverageSeq - 1; t >= 0; t--) {
			association[i].add(j);

			switch (matriceChoix[i][j]) {
			case DIAGONALE:
				i = i - 1;
				j = j - 1;
				break;
			case GAUCHE:
				j = j - 1;
				break;
			case HAUT:
				i = i - 1;
				break;
			}

		}
		return association;
	}
	
	protected synchronized ArrayList<Itemset>[] computeAssociations(
			final SymbolicSequence... tabSequence) {
		@SuppressWarnings("unchecked")
		final ArrayList<Itemset>[] tupleAssociation = new ArrayList[this
				.getNbTuples()];
		for (int i = 0; i < tupleAssociation.length; i++) {
			tupleAssociation[i] = new ArrayList<Itemset>(tabSequence.length);
		}
		int nbTuplesAverageSeq, i, j, indiceRes;
		double res = 0.0;
		final int tailleCenter = this.getNbTuples();
		int tailleT;

		for (final SymbolicSequence S : tabSequence) {

			tailleT = S.getNbTuples();

			SymbolicSequence.matriceW[0][0] = this.sequence[0]
					.squaredDistance(S.sequence[0]);
			SymbolicSequence.matriceChoix[0][0] = SymbolicSequence.RIEN;
			SymbolicSequence.optimalPathLength[0][0] = 0;

			for (i = 1; i < tailleCenter; i++) {
				SymbolicSequence.matriceW[i][0] = SymbolicSequence.matriceW[i - 1][0]
						+ this.sequence[i].squaredDistance(S.sequence[0]);
				SymbolicSequence.matriceChoix[i][0] = SymbolicSequence.HAUT;
				SymbolicSequence.optimalPathLength[i][0] = i;
			}
			for (j = 1; j < tailleT; j++) {
				SymbolicSequence.matriceW[0][j] = SymbolicSequence.matriceW[0][j - 1]
						+ S.sequence[j].squaredDistance(this.sequence[0]);
				SymbolicSequence.matriceChoix[0][j] = SymbolicSequence.GAUCHE;
				SymbolicSequence.optimalPathLength[0][j] = j;
			}

			for (i = 1; i < tailleCenter; i++) {
				for (j = 1; j < tailleT; j++) {
					indiceRes = Tools.ArgMin3(
							SymbolicSequence.matriceW[i - 1][j - 1],
							SymbolicSequence.matriceW[i][j - 1],
							SymbolicSequence.matriceW[i - 1][j]);
					SymbolicSequence.matriceChoix[i][j] = indiceRes;
					switch (indiceRes) {
					case DIAGONALE:
						res = SymbolicSequence.matriceW[i - 1][j - 1];
						SymbolicSequence.optimalPathLength[i][j] = SymbolicSequence.optimalPathLength[i - 1][j - 1] + 1;
						break;
					case GAUCHE:
						res = SymbolicSequence.matriceW[i][j - 1];
						SymbolicSequence.optimalPathLength[i][j] = SymbolicSequence.optimalPathLength[i][j - 1] + 1;
						break;
					case HAUT:
						res = SymbolicSequence.matriceW[i - 1][j];
						SymbolicSequence.optimalPathLength[i][j] = SymbolicSequence.optimalPathLength[i - 1][j] + 1;
						break;
					}
					SymbolicSequence.matriceW[i][j] = res
							+ this.sequence[i].squaredDistance(S.sequence[j]);

				}
			}
			nbTuplesAverageSeq = SymbolicSequence.optimalPathLength[tailleCenter - 1][tailleT - 1] + 1;

			i = tailleCenter - 1;
			j = tailleT - 1;

			for (int t = nbTuplesAverageSeq - 1; t >= 0; t--) {
				tupleAssociation[i].add(S.sequence[j]);
				switch (SymbolicSequence.matriceChoix[i][j]) {
				case DIAGONALE:
					i = i - 1;
					j = j - 1;
					break;
				case GAUCHE:
					j = j - 1;
					break;
				case HAUT:
					i = i - 1;
					break;
				}

			}

		}
		return tupleAssociation;
	}

	protected synchronized ArrayList<Itemset>[][] computeAssociationsBySequence(
			final SymbolicSequence... tabSequence) {
		@SuppressWarnings("unchecked")
		final ArrayList<Itemset>[][] tupleAssociation = new ArrayList[sequence.length][tabSequence.length];
		for (int i = 0; i < tupleAssociation.length; i++) {
			for (int j = 0; j < tupleAssociation[i].length; j++) {
				tupleAssociation[i][j] = new ArrayList<Itemset>();
			}
		}
		int nbTuplesAverageSeq, i, j, indiceRes;
		double res = 0.0;
		final int sequenceLength = this.sequence.length;
		int tailleT;

		for (int s = 0; s < tabSequence.length; s++) {
			final SymbolicSequence S = tabSequence[s];

			tailleT = S.getNbTuples();

			SymbolicSequence.matriceW[0][0] = this.sequence[0]
					.squaredDistance(S.sequence[0]);
			SymbolicSequence.matriceChoix[0][0] = SymbolicSequence.RIEN;
			SymbolicSequence.optimalPathLength[0][0] = 0;

			for (i = 1; i < sequenceLength; i++) {
				SymbolicSequence.matriceW[i][0] = SymbolicSequence.matriceW[i - 1][0]
						+ this.sequence[i].squaredDistance(S.sequence[0]);
				SymbolicSequence.matriceChoix[i][0] = SymbolicSequence.HAUT;
				SymbolicSequence.optimalPathLength[i][0] = i;
			}
			for (j = 1; j < tailleT; j++) {
				SymbolicSequence.matriceW[0][j] = SymbolicSequence.matriceW[0][j - 1]
						+ S.sequence[j].squaredDistance(this.sequence[0]);
				SymbolicSequence.matriceChoix[0][j] = SymbolicSequence.GAUCHE;
				SymbolicSequence.optimalPathLength[0][j] = j;
			}

			for (i = 1; i < sequenceLength; i++) {
				for (j = 1; j < tailleT; j++) {
					indiceRes = Tools.ArgMin3(
							SymbolicSequence.matriceW[i - 1][j - 1],
							SymbolicSequence.matriceW[i][j - 1],
							SymbolicSequence.matriceW[i - 1][j]);
					SymbolicSequence.matriceChoix[i][j] = indiceRes;
					switch (indiceRes) {
					case DIAGONALE:
						res = SymbolicSequence.matriceW[i - 1][j - 1];
						SymbolicSequence.optimalPathLength[i][j] = SymbolicSequence.optimalPathLength[i - 1][j - 1] + 1;
						break;
					case GAUCHE:
						res = SymbolicSequence.matriceW[i][j - 1];
						SymbolicSequence.optimalPathLength[i][j] = SymbolicSequence.optimalPathLength[i][j - 1] + 1;
						break;
					case HAUT:
						res = SymbolicSequence.matriceW[i - 1][j];
						SymbolicSequence.optimalPathLength[i][j] = SymbolicSequence.optimalPathLength[i - 1][j] + 1;
						break;
					}
					SymbolicSequence.matriceW[i][j] = res
							+ this.sequence[i].squaredDistance(S.sequence[j]);

				}
			}
			
			nbTuplesAverageSeq = SymbolicSequence.optimalPathLength[sequenceLength - 1][tailleT - 1] + 1;

			i = sequenceLength - 1;
			j = tailleT - 1;

			for (int t = nbTuplesAverageSeq - 1; t >= 0; t--) {
				tupleAssociation[i][s].add(S.sequence[j]);
				switch (SymbolicSequence.matriceChoix[i][j]) {
				case DIAGONALE:
					i = i - 1;
					j = j - 1;
					break;
				case GAUCHE:
					j = j - 1;
					break;
				case HAUT:
					i = i - 1;
					break;
				}

			}

		}

		return tupleAssociation;
	}

	@Override
	public String toString() {
		String str = "[";
		for (final Itemset t : sequence) {
			str += "{";
			str += t.toString();
			str += "}";
		}
		str += "]";
		return str;
	}

	public Itemset[] getSequence() {
		return this.sequence;
	}

	public static final double squaredL2(double a, double b) {
		double tmp = a - b;
		return tmp * tmp;
	}

	public static final double squaredL2(String a, String b) {
		return (a.equals(b)) ? 0.0 : 1.0;
	}

	public static MonoItemSet[] buildSeq(String s) {
		MonoItemSet[] seq = new MonoItemSet[s.length()];
		for (int i = 0; i < seq.length; i++) {
			seq[i] = new MonoItemSet(s.charAt(i) + "");
		}
		return seq;
	}
}
