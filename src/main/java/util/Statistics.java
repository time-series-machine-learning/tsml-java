package util;

import java.util.Arrays;
import java.util.List;

public class Statistics {

	public static double sum(double[] list) {
		double sum = 0;
		for (int i = 0; i < list.length; i++) {
			sum += list[i];
		}
		return sum;
	}	
	
	public static long sum(long[] list) {
		long sum = 0;
		for (int i = 0; i < list.length; i++) {
			sum += list[i];
		}
		return sum;
	}	
	
	public static int sum(int[] list) {
		int sum = 0;
		for (int i = 0; i < list.length; i++) {
			sum += list[i];
		}
		return sum;
	}		
	
	public static int sumIntList(List<Integer> list) {
		int sum = 0;
		int length = list.size();
		for (int i = 0; i < length; i++) {
			sum +=list.get(i);
		}
		return sum;
	}

	public static double sumDoubleList(List<Double> list) {
		double sum = 0;
		int length = list.size();
		for (int i = 0; i < length; i++) {
			sum += list.get(i);
		}
		return sum;
	}
	
	public static double meanIntList(List<Integer> list) {
		return Statistics.sumIntList(list) / list.size();
	}	
	
	public static double meanDoubleList(List<Double> list) {
		return Statistics.sumDoubleList(list) / list.size();
	}	
	
	public static double mean(double[] list) {
		double sum = 0;
		for (int i = 0; i < list.length; i++) {
			sum += list[i];
		}
		return sum/list.length;
	}
	
	public static double mean(int[] list) {
		int sum = 0;
		for (int i = 0; i < list.length; i++) {
			sum += list[i];
		}
		return sum/list.length;
	}
	
	public static double mean(long[] list) {
		long sum = 0;
		for (int i = 0; i < list.length; i++) {
			sum += list[i];
		}
		return sum/list.length;
	}
	
	public static double standard_deviation_population(double[] list) {
		double mean = Statistics.mean(list);
		double sq_sum = 0;
		
		for (int i = 0; i < list.length; i++) {
			sq_sum += (list[i] - mean) * (list[i] - mean);
		}
		
		return Math.sqrt(sq_sum/list.length);
	}
	
	public static double standard_deviation_population(long[] list) {
		double mean = Statistics.mean(list);
		long sq_sum = 0;
		
		for (int i = 0; i < list.length; i++) {
			sq_sum += (list[i] - mean) * (list[i] - mean);
		}
		
		return Math.sqrt((double)sq_sum/list.length);
	}

	public static double standard_deviation_population(int[] list) {
		double mean = Statistics.mean(list);
		long sq_sum = 0;
		
		for (int i = 0; i < list.length; i++) {
			sq_sum += (list[i] - mean) * (list[i] - mean);
		}
		
		return Math.sqrt((double)sq_sum/list.length);
	}
	
	public static double median(double[] numArray) {
		Arrays.sort(numArray);
		double median;
		if (numArray.length % 2 == 0)
		    median = ((double)numArray[numArray.length/2] + (double)numArray[numArray.length/2 - 1])/2;
		else
		    median = (double) numArray[numArray.length/2];		
		return median;
	}
	
}
