package core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import org.apache.commons.lang3.time.DurationFormatUtils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import trees.ProximityForest;
import trees.ProximityTree;
import util.Statistics;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class ProximityForestResult {
	
	private transient ProximityForest forest;
	public boolean results_collated = false;
	
	//FILLED BY FOREST CLASS
	public int forest_id = -1;
	public int majority_vote_match_count = 0;

	public long startTimeTrain = 0;
	public long endTimeTrain = 0;
	public long elapsedTimeTrain = 0;
	
	public long startTimeTest = 0;
	public long endTimeTest = 0;
	public long elapsedTimeTest = 0;	
			
	public int errors = 0, correct = 0;	
	public double accuracy = 0, error_rate = 0;	
	
	
	//FILLED BY STAT COLLECTOR CLASS

	public int total_num_trees = -1;

	//num nodes
	public double mean_num_nodes_per_tree = -1;
	public double sd_num_nodes_per_tree = -1;
//	public int min_num_nodes_per_tree = -1;
//	public int max_num_nodes_per_tree = -1;

	//depth
	public double mean_depth_per_tree = -1;
	public double sd_depth_per_tree = -1;
//	public int min_depth_per_tree = -1;
//	public int max_depth_per_tree = -1;	
	

	//weighted depth //TODO comment add formula
	public double mean_weighted_depth_per_tree = -1;
	public double sd_weighted_depth_per_tree = -1;
//	public int min_weighted_depth_per_tree = -1;
//	public int max_weighted_depth_per_tree = -1;	
	
	//memory
//	int memory_usage = 0;
	
	//distance timings
//	public long dtw_time;
//	public long dtwcv_time;
//	public long ddtw_time;
//	public long ddtwcv_time;
//	public long wtw_time;
//	public long wddtw_time;	
//	public long euc_time;
//	public long lcss_time;
//	public long msm_time;	
//	public long twe_time;
//	public long erp_time;	
	
	//call counts
//	public long dtw_count;
//	public long dtwcv_count;
//	public long ddtw_count;
//	public long ddtwcv_count;
//	public long wtw_count;
//	public long wddtw_count;	
//	public long euc_count;
//	public long lcss_count;
//	public long msm_count;	
//	public long twe_count;
//	public long erp_count;	
	
	public ProximityForestResult(ProximityForest forest) {
		this.forest_id = forest.getForestID();
		this.forest = forest;
	}
	
	public void collateResults() {
		
		if (results_collated) {
			return;
		}		
		
		ProximityTree[] trees = forest.getTrees();
		ProximityTree tree;
		TreeStatCollector tree_stats;
		
		total_num_trees = trees.length;
		
		int nodes[] = new int[total_num_trees];
		double depths[] = new double[total_num_trees];
		double weighted_depths[] = new double[total_num_trees];
				
		for (int i = 0; i < total_num_trees; i++) {
			tree = trees[i];
			tree_stats = tree.getTreeStatCollection();
			
			nodes[i] = tree_stats.num_nodes;
			depths[i] = tree_stats.depth;
			weighted_depths[i] = tree_stats.weighted_depth;
			
		}
		mean_num_nodes_per_tree = Statistics.mean(nodes);
		sd_num_nodes_per_tree = Statistics.standard_deviation_population(nodes);
		
		mean_depth_per_tree = Statistics.mean(depths);
		sd_depth_per_tree = Statistics.standard_deviation_population(depths);
		
		mean_weighted_depth_per_tree = Statistics.mean(weighted_depths);
		sd_weighted_depth_per_tree = Statistics.standard_deviation_population(weighted_depths);
		
		
		results_collated = true;
	}
	
	public void printResults(String datasetName, int experiment_id, String prefix) {
		
//		System.out.println(prefix+ "-----------------Experiment No: " 
//				+ experiment_id + " (" +datasetName+ "), Forest No: " 
//				+ (this.forest_id) +"  -----------------");
		
		if (AppContext.verbosity > 0) {
			String time_duration = DurationFormatUtils.formatDuration((long) (elapsedTimeTrain/1e6), "H:m:s.SSS");
	        System.out.format("%sTraining Time: %fms (%s)\n",prefix, elapsedTimeTrain/1e6, time_duration);
			time_duration = DurationFormatUtils.formatDuration((long) (elapsedTimeTest/1e6), "H:m:s.SSS");		
	        System.out.format("%sPrediction Time: %fms (%s)\n",prefix, elapsedTimeTest/1e6, time_duration);
	
	        
	        System.out.format("%sCorrect(TP+TN): %d vs Incorrect(FP+FN): %d\n",prefix,  correct, errors);
	        System.out.println(prefix+"Accuracy: " + accuracy);
	        System.out.println(prefix+"Error Rate: "+ error_rate);			
		}

        
        this.collateResults();
        
        //this is just the same info in a single line, used to grep from output and save to a csv, use the #: marker to find the line easily
       
        //the prefix REPEAT is added to this comma separated line easily use grep from command line to filter outputs to a csv file
        //just a quick method to filter important info while in command line
        
        String pre = "REPEAT:" + (experiment_id+1) +" ,";
		System.out.print(pre + datasetName);        
		System.out.print(", " + accuracy);
		System.out.print(", " + elapsedTimeTrain /1e6);
		System.out.print(", " + elapsedTimeTest /1e6);
		System.out.print(", " + mean_depth_per_tree);
//		System.out.print(", " + mean_weighted_depth_per_tree);
		System.out.println();
	}
	
	public String exportJSON(String datasetName, int experiment_id) throws Exception {
		String file = "";
		String timestamp = LocalDateTime.now()
			       .format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS"));		
		
		file = AppContext.output_dir + File.separator + forest_id + timestamp;
		
		File fileObj = new File(file);
		
		fileObj.getParentFile().mkdirs();
		fileObj.createNewFile();		
		
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))){		
		
			Gson gson;
			GsonBuilder gb = new GsonBuilder();
			gb.serializeSpecialFloatingPointValues();
			gb.serializeNulls();
			gson = gb.create();
			
//			SerializableResultSet object = new SerializableResultSet(this.forests);
			
			bw.write(gson.toJson(this));
			bw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}finally {
//			bw.close();
		}
					
		return file;
	}
	
}
