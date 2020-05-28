package core;

import java.util.Random;

import core.contracts.Dataset;
import distance.elastic.MEASURE;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class AppContext {
	
	private static final long serialVersionUID = -502980220452234173L;
	public static final String version = "1.0.0";
	
	public static final int ONE_MB = 1048576;	
	public static final String TIMESTAMP_FORMAT_LONG = "yyyy-MM-dd HH:mm:ss.SSS";	
	public static final String TIMESTAMP_FORMAT_SHORT = "HH:mm:ss.SSS";	
	
	
	//********************************************************************
	//DEVELOPMENT and TESTING AREA -- 
	public static boolean config_majority_vote_tie_break_randomly = true;
	public static boolean config_skip_distance_when_exemplar_matches_query = true;
	public static boolean config_use_random_choice_when_min_distance_is_equal = true;	
	//********************************************************************
	
	//DEFAULT SETTINGS, these are overridden by command line arguments
	public static long rand_seed;	//TODO set seed to reproduce results
	public static Random rand;
	
	public static int verbosity = 0; //0, 1, 2 
	public static int export_level = 1; //0, 1, 2 

	public static String training_file = "E:/data/ucr/cleaned/ItalyPowerDemand/ItalyPowerDemand_TRAIN.csv";
	public static String testing_file = "E:/data/ucr/cleaned/ItalyPowerDemand/ItalyPowerDemand_TEST.csv";
	public static String output_dir = "output/";
	public static boolean csv_has_header = false;
	public static boolean target_column_is_first = true;


	public static int num_repeats = 1;
	public static int num_trees = 1;
	public static int num_candidates_per_split = 1;
	public static boolean random_dm_per_node = true;
	public static boolean shuffle_dataset = false;
		
	public static boolean warmup_java = false;
	public static boolean garbage_collect_after_each_repetition = true;	
	
	public static int print_test_progress_for_each_instances = 100;
	
	public static MEASURE[] enabled_distance_measures = new MEASURE[] {
			MEASURE.euclidean,
			MEASURE.dtw,
			MEASURE.dtwcv,
			MEASURE.ddtw,
			MEASURE.ddtwcv,
			MEASURE.wdtw,
			MEASURE.wddtw,
			MEASURE.lcss,
			MEASURE.erp,
			MEASURE.twe,
			MEASURE.msm	
	};	

	public static Runtime runtime = Runtime.getRuntime();	
	
	private static transient Dataset train_data;
	private static transient Dataset test_data;
	private static String datasetName; 
	
	static {
		rand = new Random();
	}

	public static Random getRand() {
		return rand;
	}

	public static Dataset getTraining_data() {
		return train_data;
	}

	public static void setTraining_data(Dataset train_data) {
		AppContext.train_data = train_data;
	}

	public static Dataset getTesting_data() {
		return test_data;
	}

	public static void setTesting_data(Dataset test_data) {
		AppContext.test_data = test_data;
	}

	public static String getDatasetName() {
		return datasetName;
	}

	public static void setDatasetName(String datasetName) {
		AppContext.datasetName = datasetName;
	}
}
