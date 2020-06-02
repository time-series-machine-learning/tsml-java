package util;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import core.contracts.Dataset;
import datasets.ListDataset;

public class Sampler {
	
	private static Random rand = new Random();
	
	public Sampler(Random rand) {
	
//		this.rand = rand;
	}
	
	
//reference
//https://stackoverflow.com/questions/4702036/take-n-random-elements-from-a-liste?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	
	public static <E> List<E> pickNRandomElements(List<E> list, int n, Random r) {
	    int length = list.size();

	    if (length < n) return null;

	    //We don't need to shuffle the whole list
	    for (int i = length - 1; i >= length - n; --i)
	    {
	        Collections.swap(list, i , r.nextInt(i + 1));
	    }
	    return list.subList(length - n, length);
	}

	public static <E> List<E> pickNRandomElements(List<E> list, int n) {
	    return pickNRandomElements(list, n, ThreadLocalRandom.current());
	}	
	
	
	/**
	 * An improved version (Durstenfeld) of the Fisher-Yates algorithm with O(n) time complexity
	 * Permutes the given array
	 * @param array array to be shuffled
	 * reference
	 * @url http://www.programming-algorithms.net/article/43676/Fisher-Yates-shuffle
	 */
	public static void fisherYatesKnuthShuffle(int[] array) {
//	    Random r = new Random();
	    for (int i = array.length - 1; i > 0; i--) {
	        int index = rand.nextInt(i);
	        //swap
	        int tmp = array[index];
	        array[index] = array[i];
	        array[i] = tmp;
	    }
	} 	
	
	public static ListDataset uniform_sample(Dataset dataset, int n) {
		
		n = n > dataset.size() ? dataset.size() : n;
		
		ListDataset sample = new ListDataset(n, dataset.length());
		
		int[] indices = new int[n];
		for (int i = 0; i < n; i++) {
			indices[i] = i;
		}
		Sampler.fisherYatesKnuthShuffle(indices);
	
		for (int i = 0; i < n; i++) {
			sample.add(dataset.get_class(i), dataset.get_series(i));
		}
		
		return sample;
	}
	
	//TODO naive implementation, quick fix
	public static ListDataset uniform_sample(Dataset dataset, int n, double[][] exclude) {
		ListDataset sample = Sampler.uniform_sample(dataset, n);
		int size = sample.size();
		
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < exclude.length; j++) {
				if (sample.get_series(i) == exclude[j]) {
					sample.remove(i);
				}
			}
		}
		
		return sample;
	}
	
	public static ListDataset stratified_sample(Map<Integer, ListDataset> data_per_class, 
			int n_per_class, boolean shuffle, double[][] exclude) {
		ListDataset sample = new ListDataset(data_per_class.size() * n_per_class);
		ListDataset class_sample;
		int class_sample_size;
		
		for (Map.Entry<Integer, ListDataset> entry : data_per_class.entrySet()) {
			
			if (exclude == null) {
				class_sample = Sampler.uniform_sample(entry.getValue(), n_per_class);
			}else {
				class_sample = Sampler.uniform_sample(entry.getValue(), n_per_class, exclude);
			}
			class_sample_size = class_sample.size();

			for (int i = 0; i < class_sample_size; i++) {
				sample.add(class_sample.get_class(i), class_sample.get_series(i));
			}
		}
		
		if (shuffle) {
			sample.shuffle();
		}		
		
		return sample;
	}
	
	public static Map<Integer, ListDataset> stratified_sample_per_class(
			Map<Integer, ListDataset> data_per_class, int n_per_class, 
			boolean shuffle_each_class, double[][] exclude) {
		Map<Integer, ListDataset> sample = new HashMap<Integer, ListDataset> ();
		ListDataset class_sample;
		
		for (Map.Entry<Integer, ListDataset> entry : data_per_class.entrySet()) {
			if (exclude == null) {
				class_sample = Sampler.uniform_sample(entry.getValue(), n_per_class);
			}else {
				class_sample = Sampler.uniform_sample(entry.getValue(), n_per_class, exclude);
			}			
			
			if (shuffle_each_class) {
				class_sample.shuffle();
			}	
			
			sample.put(entry.getKey(), class_sample);
		}
		
		return sample;
	}		
}
