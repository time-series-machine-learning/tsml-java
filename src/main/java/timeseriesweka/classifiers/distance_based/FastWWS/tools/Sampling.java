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

import java.util.Random;

import weka.core.Instances;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * Different types of sampling for the datasets
 * 
 * @author Chang Wei Tan
 *
 */
public class Sampling {
	/**
	 * Sample a subset from train and test set each
	 * @param train
	 * @param numTrain
	 * @param test
	 * @param numTest
	 * @return
	 */
	public static Instances[] sample(Instances train, int numTrain, Instances test, int numTest) {		
		Instances trainDataset = new Instances(train,numTrain);
		trainDataset = random(train);
		trainDataset = new Instances(trainDataset, 0, numTrain);
		
		Instances testDataset = new Instances(test,numTest);
		testDataset = random(test);
		testDataset = new Instances(testDataset, 0, numTest);
		
		return new Instances[] { trainDataset, testDataset };
	}
	
	/** 
	 * Sample a subset from the given dataset
	 * @param data
	 * @param size
	 * @return
	 */
	public static Instances sample(Instances data, int size) {
		Instances newData = new Instances(data,size);
		newData = random(data);
		newData = new Instances(newData, 0, size);
		
		return newData;
	}
	
	/** 
	 * Randomize the dataset
	 * @param data
	 * @return
	 */
	public static Instances random(Instances data) {
		data.randomize(new Random());
		return data;
	}
	
	/** 
	 * Reorder the dataset by its largest class
	 * @param data
	 * @return
	 */
	public static Instances orderByLargestClass(Instances data) {
		Instances newData = new Instances(data, data.numInstances());
		
		// get the number of class in the data
		int nbClass = data.numClasses();
		int[] instancePerClass = new int[nbClass];
		int[] labels = new int[nbClass];
		int[] classIndex = new int[nbClass];
		
		// sort the data base on its class
		data.sort(data.classAttribute());
		
		// get the number of instances per class in the data
		for (int i = 0; i < nbClass; i++) {
			instancePerClass[i] = data.attributeStats(data.classIndex()).nominalCounts[i];
			labels[i] = i;
			if (i > 0)
				classIndex[i] = classIndex[i-1] + instancePerClass[i-1];
		}
		QuickSort.sort(instancePerClass, labels);
		
		for (int i = nbClass-1; i >=0 ; i--) {
			for (int j = 0; j < instancePerClass[i]; j++) {
				newData.add(data.instance(classIndex[labels[i]] + j));
			}
		}
		
		return newData;
	}
	
	/** 
	 * Reorder the data by compactness of each class using Euclidean distance
	 * @param data
	 * @return
	 */
	public static Instances orderByCompactClass(Instances data) {
		Instances newData = new Instances(data, data.numInstances());
		
		// get the number of class in the data
		int nbClass = data.numClasses();
		int[] instancePerClass = new int[nbClass];
		int[] labels = new int[nbClass];
		int[] classIndex = new int[nbClass];
		double[] compactness = new double[nbClass];
		
		// sort the data base on its class
		data.sort(data.classAttribute());
		
		int start = 0;
		// get the number of instances per class in the data
		for (int i = 0; i < nbClass; i++) {
			instancePerClass[i] = data.attributeStats(data.classIndex()).nominalCounts[i];
			labels[i] = i;
			if (i > 0) 
				classIndex[i] = classIndex[i-1] + instancePerClass[i-1];
			int end = start + instancePerClass[i];
			int counter = 0;
			double[][] dataPerClass = new double[instancePerClass[i]][data.numAttributes()-1];
			for (int j = start; j < end; j++) {
				dataPerClass[counter++] = data.instance(j).toDoubleArray();
			}
			double[] mean = arithmeticMean(dataPerClass);
			double d = 0;
			for (int j = 0; j < instancePerClass[i]; j++) {
				double temp = euclideanDistance(mean, dataPerClass[j]);
				temp *= temp;
				temp -= (mean[0] - dataPerClass[j][0]) * (mean[0] - dataPerClass[j][0]);
				d += temp;
			}
			compactness[i] = d / instancePerClass[i];
			start = end;
		}
		
		QuickSort.sort(compactness, labels);
		
		for (int i = nbClass-1; i >=0 ; i--) {
			for (int j = 0; j < instancePerClass[labels[i]]; j++) {
				newData.add(data.instance(classIndex[labels[i]] + j));
			}
		}
		
		return newData;
	}
	
	/** 
	 * Compute Euclidean distance between two sequences
	 * @param x
	 * @param y
	 * @return
	 */
	private static double euclideanDistance(double[] x, double[] y) {
		double dist = 0;
		for (int i = 0; i < x.length; i++) {
			dist += (x[i]-y[i]) * (x[i]-y[i]);
		}
		
		return Math.sqrt(dist);
	}
	
	/**
	 * Compute mean of a set of sequences
	 * @param array
	 * @return
	 */
	public static double[] arithmeticMean(double[][] array) {
		double[] mean = new double[array[0].length];
		for (int i = 0; i < array[0].length; i++) {
			for (int j = 0; j < array.length; j++) {
				mean[i] += array[j][i];
			}
			mean[i]/=array.length;
		}
		
		return mean;
	}
}
