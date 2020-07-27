package trees;

import java.util.Map;

import core.AppContext;
import core.contracts.Dataset;
import datasets.ListDataset;
import distance.elastic.DistanceMeasure;
import utilities.Utilities;
import weka.core.Instances;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class Splitter{
	
	protected int num_children; //may be move to splitter?
	protected DistanceMeasure distance_measure;
	protected double[][] exemplars;
	
	protected DistanceMeasure temp_distance_measure;
	protected double[][] temp_exemplars;	
	
	ListDataset[] best_split = null;	
	ProximityTree.Node node;
	
	public Splitter(ProximityTree.Node node) throws Exception {
		this.node = node;	
	}
	
	public ListDataset[] split_data(Dataset sample, Map<Integer, ListDataset> data_per_class) throws Exception {
//		num_children = sample.get_num_classes();
		ListDataset[] splits = new ListDataset[sample.get_num_classes()];
		temp_exemplars = new double[sample.get_num_classes()][];

		int branch = 0;
		for (Map.Entry<Integer, ListDataset> entry : data_per_class.entrySet()) {
			int r = AppContext.getRand().nextInt(entry.getValue().size());
			
			splits[branch] = new ListDataset(sample.size(), sample.length());
			//use key just in case iteration order is not consistent
			temp_exemplars[branch] = entry.getValue().get_series(r);
			branch++;
		}
		
		int sample_size = sample.size();
		int closest_branch = -1;
		for (int j = 0; j < sample_size; j++) {
			closest_branch = this.find_closest_branch(sample.get_series(j), 
					temp_distance_measure, temp_exemplars);

//			System.out.println("cb: " + j + "," + closest_branch); // todo

			if (closest_branch == -1) {
				assert false;
			}
			splits[closest_branch].add(sample.get_class(j), sample.get_series(j));
		}

		return splits;
	}	

	public int find_closest_branch(double[] query, DistanceMeasure dm, double[][] e) throws Exception{
		return dm.find_closest_node(query, e, true);
	}	
	
	public int find_closest_branch(double[] query) throws Exception{
		return this.distance_measure.find_closest_node(query, exemplars, true);
	}		
	
	public Dataset[] getBestSplits() {
		return this.best_split;
	}
	
	public ListDataset[] find_best_split(Dataset data) throws Exception {

		Map<Integer, ListDataset> data_per_class = data.split_classes();
		
		double weighted_gini = Double.POSITIVE_INFINITY;
		double best_weighted_gini = Double.POSITIVE_INFINITY;
		ListDataset[] splits = null;
		int parent_size = data.size();
	
		for (int i = 0; i < AppContext.num_candidates_per_split; i++) {
//			System.out.println("pd");
			if (AppContext.random_dm_per_node) {
				int r = AppContext.getRand().nextInt(AppContext.enabled_distance_measures.length);
				temp_distance_measure = new DistanceMeasure(AppContext.enabled_distance_measures[r]);		
			}else {
				//NOTE: num_candidates_per_split has no effect if random_dm_per_node == false (if DM is selected once per tree)
				//after experiments we found that DM selection per node is better since it diversifies the ensemble
				temp_distance_measure = node.tree.tree_distance_measure;
			}
			
			temp_distance_measure.select_random_params(data, AppContext.getRand());

//			System.out.println("pe");
//			System.out.println("is: " + data.size());
			for(Map.Entry<Integer, ListDataset> entry : data_per_class.entrySet()) {
//				System.out.println("cc: " + entry.getValue().size());
			}
			splits = split_data(data, data_per_class);
			weighted_gini = weighted_gini(parent_size, splits);
//			System.out.println("g: " + Utilities.roundExact(weighted_gini, 8));

			if (weighted_gini <  best_weighted_gini) {
				best_weighted_gini = weighted_gini;
				best_split = splits;
				distance_measure = temp_distance_measure;
				exemplars = temp_exemplars;
			}
		}

//		System.out.println("bg: " + Utilities.roundExact(best_weighted_gini, 8));

//		if(best_weighted_gini == 0.26666666666666666) {
//			System.out.println("stop here");
//		}

		this.num_children = best_split.length;


		for(ListDataset part : best_split) {
//			System.out.println("part: " + part.size());
		}

		return this.best_split;
	}
	
	public double weighted_gini(int parent_size, ListDataset[] splits) {
		double wgini = 0.0;
		
		for (int i = 0; i < splits.length; i++) {
			wgini = wgini + ((double) splits[i].size() / parent_size) * splits[i].gini();
		}

		return wgini;
	}	
	
}
