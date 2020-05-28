package datasets;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import java.util.Set;
import core.contracts.Dataset;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class ListDataset implements Dataset{

	private List<double[]> data;
	private List<Integer> labels;
	private Map<Integer, Integer> class_map;	//key = class label, value = class size
	private Map<Integer, Integer> initial_class_labels;	//key = old label, value = new label
	private int length;
	private boolean is_reodered = false;
		
	public ListDataset() {
		this.data = new ArrayList<double[]>();
		this.labels = new ArrayList<Integer>();
		this.class_map = new LinkedHashMap<Integer, Integer>();
	}
	
	public ListDataset(int expected_size) {
		this.data = new ArrayList<double[]>(expected_size);
		this.labels = new ArrayList<Integer>(expected_size);
		this.class_map = new LinkedHashMap<Integer, Integer>();
	}
	
	public ListDataset(int expected_size, int length) {
		this.length = length;
		this.data = new ArrayList<double[]>(expected_size);
		this.labels = new ArrayList<Integer>(expected_size);
		this.class_map = new LinkedHashMap<Integer, Integer>();
	}	
	
	public int size() {
		return this.data.size();
	}
	
	public int length() {
		//Note length for empty dataset may result in either 0, or length used in the constructor
		return this.data.isEmpty() ? length : this.data.get(0).length;
	}	
	
	public void add(Integer label, double[] series) {
		//TODO length consistency check not done here
		
		this.data.add(series);
		this.labels.add(label);
		
		if (class_map.containsKey(label)) {
			class_map.put(label, class_map.get(label) + 1);
		}else {
			class_map.put(label, 1);
		}
		
	}
	
	public void remove(int i) {
		Integer label = this.labels.get(i);
		
		if (class_map.containsKey(label)) {
			int count = class_map.get(label);
			if (count > 0) {
				class_map.put(label, class_map.get(label) - 1);
			}else {
				class_map.remove(label);
			}
		}
		
		this.data.remove(i);
		this.labels.remove(i);
	}
	
	public double[] get_series(int i) {
		return this.data.get(i);
	}

	public Integer get_class(int i) {
		return this.labels.get(i);
	}
	
	public int get_num_classes() {
		return this.class_map.size();
	}
	
	public int get_class_size(Integer label) {
		return this.class_map.get(label);
	}	
	
	public Map<Integer, Integer> get_class_map() {
		return this.class_map;
	}	
	
	//TODO if ordered from zero, theres a faster way to do this, just return a continuous list
	public int[] get_unique_classes() {
		Set<Integer> keys = this.get_class_map().keySet();
		int[] unqique_classes = new int[keys.size()];
		int i = 0;
		
		for (Integer integer : keys) {
			unqique_classes[i] = integer;
			i++;
		}
		
		return unqique_classes;
	}	
	
	public Set<Integer> get_unique_classes_as_set() {
		return this.get_class_map().keySet();
	}		
	
	public Map<Integer, ListDataset> split_classes() {
		Map<Integer, ListDataset> split = new LinkedHashMap<Integer, ListDataset>();
		Integer label;
		ListDataset class_set = null;
		int size = this.size();
		
		for (int i = 0; i < size; i++) {
			label = this.labels.get(i);
			if (! split.containsKey(label)) {
				class_set = new ListDataset(this.class_map.get(label));
				split.put(label, class_set);
			}
			
			split.get(label).add(label, this.data.get(i));
		}
		
		return split;
	}
	
//	public Map<Integer, DatasetIndex> get_classes_as_indices() {
//		Map<Integer, DatasetIndex> class_map = new LinkedHashMap<Integer, DatasetIndex>();
//		Integer label;
//		DatasetIndex class_set = null;
//		
//		for (int i = 0; i < this.data.size(); i++) {
//			label = this.labels.get(i);
//			if (! class_map.containsKey(label)) {
//				class_set = new DatasetIndex(this, this.class_sizes.get(label));
//				class_map.put(label, class_set);
//			}
//			
//			class_map.get(label).add(label, i);
//		}
//		
//		return class_map;
//	}	

	
	public double gini() {
		double sum = 0;
		double p;
		int total_size = this.data.size();
		
		for (Entry<Integer, Integer> entry: class_map.entrySet()) {
			p = (double)entry.getValue() / total_size;
			sum += p * p;
		}
		
		return 1 - sum;
	}
	
	public List<double[]> _internal_data_list() {
		return this.data;
	}	
	
	public List<Integer> _internal_class_list() {
		return this.labels;
	}
	
	public double[][] _internal_data_array(){
		//not implemented for this class
		return null;
	}
	
	public int[] _internal_class_array() {
		//not implemented for this class
		return null;
	}
		
	public ListDataset reorder_class_labels(Map<Integer, Integer> new_order) {
		ListDataset new_dataset = new ListDataset(this.size(), this.length());
		//key = old label, value = new label, easier to build this way, later we swap to new=>old
		if (new_order == null) {
			new_order = new HashMap<Integer, Integer>();
		}
		
		int size = this.size();
		Integer old_label;
		int new_label = 0;
		int temp_label = 0;
		
		for (int i = 0; i < size; i++) {
			old_label = labels.get(i);
		
			if (new_order.containsKey(old_label)) {
				temp_label = new_order.get(old_label);
			}else {
				new_order.put(old_label, new_label);
				temp_label = new_label;
				new_label++;
			}
			
			new_dataset.add(temp_label, data.get(i));
		}
	
		//reverse old=>new to new=>old map, because its easier to use 
//		Map<Integer, Integer> swapped = new_order.entrySet().stream().collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));
		new_dataset.setInitialClassOrder(new_order);
		new_dataset.setReordered(true);
		return new_dataset;
	}
	
	public Map<Integer, Integer> _get_initial_class_labels(){
		return this.initial_class_labels;
	}
	
	
	public void setReordered(boolean status) {
		this.is_reodered = status;
	}
	
	public void setInitialClassOrder(Map<Integer, Integer> initial_order) {
		this.initial_class_labels = initial_order;
	}
	
	public String toString() {
		return this.class_map.toString();
	}		
	
	//refer https://stackoverflow.com/questions/13532919/how-do-i-shuffle-two-arrays-in-same-order-in-java
	//must create two instances of random number generator with same seed
	//same reusing the generator wont give same values
	//TODO warning
	//if you do very quick shufflings, then if the value return by nanoTime (which has millisecond precision)
	//this will produce same results;
	public void shuffle() {
		this.shuffle(System.nanoTime());
	}
	
	public void shuffle(long seed) {
		Collections.shuffle(data, new Random(seed));
		Collections.shuffle(labels, new Random(seed));

	}
	
	public ListDataset sample_n(int n_items, Random rand) {
		int n = n_items;
		if (n > this.size()) {
			n = this.size();
		}
		
		ListDataset sample = new ListDataset(n, this.length());
		
		this.shuffle(); //TODO changes dataset ordering
		
		for (int i = 0; i < n; i++) {
			sample.add(labels.get(i), data.get(i));
		}
		
		return sample;
	}
	
	public void swap(int from, int to) {
		double[] tmp_series;
		Integer tmp_label;
		
		tmp_series = this.data.get(to);
		tmp_label = this.labels.get(to);
		
		this.data.set(to, this.data.get(from));
		this.labels.set(to, this.labels.get(from));
		
		this.data.set(from, tmp_series);
		this.labels.set(from, tmp_label);
	}

	@Override
	public ListDataset shallow_clone() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public ListDataset deep_clone() {
		// TODO Auto-generated method stub
		return null;
	}	

	@Override
	public ListDataset sort_on(int timestamp) {
		// TODO Auto-generated method stub
		return null;
	}
	
}

 