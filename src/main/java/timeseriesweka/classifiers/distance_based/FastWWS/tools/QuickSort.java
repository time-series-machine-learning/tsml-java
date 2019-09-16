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
package timeseriesweka.classifiers.distance_based.FastWWS.tools;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * Performing quicksort 
 * 
 * @author Chang Wei tan
 *
 */
public class QuickSort {	
	/** 
	 * Sorting double array with index
	 * @param numbers
	 * @param index
	 */
	public final static void sort(double[] numbers, int[] index) {
		qsort(numbers, index, 0, numbers.length-1);
	}
	
	/**
	 * Sorting integer array with index
	 * @param numbers
	 * @param index
	 */
	public final static void sort(int[] numbers, int[] index) {
		qsort(numbers, index, 0, numbers.length-1);
	}
	
	/**
	 * Quicksort algorithm for double array
	 * @param numbers
	 * @param index
	 * @param low
	 * @param high
	 */
	public final static void qsort(double[] numbers, int[] index, int low, int high) {		
		int i = low, j = high;
		
		final double pivot = numbers[(low + high)/2];
		
		while (i <= j) {
			while (numbers[i] < pivot)
				i++;
			while (numbers[j] > pivot)
				j--;
			if (i <= j) {
				swap(numbers, index, i, j);
				i++;
				j--;
			}
		}
		if (low < j)
			qsort(numbers, index, low, j);
		if (i < high)
			qsort(numbers, index, i, high);
	}
	
	/**
	 * Quicksort algorithm for Integer array
	 * @param numbers
	 * @param index
	 * @param low
	 * @param high
	 */
	public final static void qsort(int[] numbers, int[] index, int low, int high) {		
		int i = low, j = high;
		
		final double pivot = numbers[(low + high)/2];
		
		while (i <= j) {
			while (numbers[i] < pivot)
				i++;
			while (numbers[j] > pivot)
				j--;
			if (i <= j) {
				swap(numbers, index, i, j);
				i++;
				j--;
			}
		}
		if (low < j)
			qsort(numbers, index, low, j);
		if (i < high)
			qsort(numbers, index, i, high);
	}
	
	/**
	 * Swap operation
	 * @param numbers
	 * @param index
	 * @param i
	 * @param j
	 */
	private final static void swap(double[] numbers, int[] index, int i, int j) {
		final double tempNum = numbers[i];
		final int tempIndex = index[i];
		numbers[i] = numbers[j];
		index[i] = index[j];
		numbers[j] = tempNum;
		index[j] = tempIndex;
	} 
	
	/**
	 * Swap operation
	 * @param numbers
	 * @param index
	 * @param i
	 * @param j
	 */
	private final static void swap(int[] numbers, int[] index, int i, int j) {
		final int tempNum = numbers[i];
		final int tempIndex = index[i];
		numbers[i] = numbers[j];
		index[i] = index[j];
		numbers[j] = tempNum;
		index[j] = tempIndex;
	} 
}
