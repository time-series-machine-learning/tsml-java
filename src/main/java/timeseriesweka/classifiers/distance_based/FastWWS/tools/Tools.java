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
package timeseriesweka.classifiers.distance_based.FastWWS.tools;

import java.util.Random;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * Some basic tools for simple operations
 * 
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 *
 */
public class Tools {
	/** 
	 * Minimum of 3 elements
	 * @param a
	 * @param b
	 * @param c
	 * @return
	 */
	public final static double Min3(final double a, final double b, final double c) {
		return (a <= b) ? ((a <= c) ? a : c) : (b <= c) ? b : c;
	}

	/**
	 * Argument for the minimum of 3 elements
	 * @param a
	 * @param b
	 * @param c
	 * @return
	 */
	public static int ArgMin3(final double a, final double b, final double c) {
		return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
	}
	
	/** 
	 * Sum of an array
	 * @param tab
	 * @return
	 */
	public static double sum(final double... tab) {
		double res = 0.0;
		for (double d : tab)
			res += d;
		return res;
	}
	
	/** 
	 * Maximum of an array
	 * @param tab
	 * @return
	 */
	public static double max(final double... tab) {
		double max = Double.NEGATIVE_INFINITY;
		for (double d : tab){
			if(max<d){
				max = d;
			}
		}
		return max;
	}
	
	/** 
	 * Minimum of an array
	 * @param tab
	 * @return
	 */
	public static double min(final double... tab) {
		double min = Double.POSITIVE_INFINITY;
		for (double d : tab){
			if(d<min){
				min = d;
			}
		}
		return min;
	}
	
	/**
	 * Generate random permutation given a length
	 * @param length
	 * @return
	 */
	public static final int[] randPermutation(int length) {
		int[] array = new int[length];
		for (int i = 0; i < length; i++) {
			array[i] = i;
		}
		return randPermutation(array);
	}
	
	/**
	 * Generate random permutation given an array
	 * @param array
	 * @return
	 */
	public static final int[] randPermutation(int[] array) {
		Random r = new Random();
		int randNum;
		for (int i = 0; i < array.length; i++) {
			randNum = i + r.nextInt(array.length-i);
			swap(array, i, randNum);
		}
		return array;
	}
	
	/**
	 * Swap operation
	 * @param array
	 * @param a
	 * @param b
	 * @return
	 */
	public static final int[] swap(int[] array, int a, int b){
		int temp = array[a];
		array[a] = array[b];
		array[b] = temp;
		return array;
	}
	
	/**
	 * Swap operation
	 * @param array
	 * @param a
	 * @param b
	 * @return
	 */
	public static final double[] swap(double[] array, int a, int b){
		double temp = array[a];
		array[a] = array[b];
		array[b] = temp;
		return array;
	}
	
	/** 
	 * Squared Euclidean distance
	 * @param a
	 * @param b
	 * @return
	 */
	public static final double squaredEuclidean(double a, double b) {
		return (a - b) * (a - b);
	}
	
}
