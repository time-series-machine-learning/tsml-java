package util;

import core.AppContext;

public class PrintUtilities {

	public static void abort(Exception e) {
//        System.out.println("\nFatal Error::" + e.getMessage() + "\n\n");
        System.err.println("\nFatal Error:: " + e.getMessage() + "\n");
        e.printStackTrace();
        System.exit(-1);
	}
	
	public static void printMemoryUsage() {
		PrintUtilities.printMemoryUsage(false);
	}	
	
	public static void printMemoryUsage(boolean minimal) {
		long avail_mem, free_mem, used_mem;
		avail_mem = AppContext.runtime.totalMemory() / AppContext.ONE_MB;
		free_mem = AppContext.runtime.freeMemory() / AppContext.ONE_MB;
		used_mem = avail_mem - free_mem;
		if (minimal) {
			System.out.print("(" + used_mem + "/" + avail_mem + "MB) ");
		}else {
			System.out.println("Using: " + used_mem + " MB, Free: " + free_mem 
					+ " MB, Allocated Pool: " + avail_mem+ " MB, Max Available: " 
					+ AppContext.runtime.maxMemory()/ AppContext.ONE_MB + " MB");				
		}

	}

	public static void printConfiguration() {
		// TODO Auto-generated method stub
		System.out.println("Running on configurations...");
		System.out.println("Dataset: " + AppContext.getDatasetName() 
		+ ", Training Data : " + AppContext.getTraining_data().size() + "x" + AppContext.getTraining_data().length()
		+ " , Testing Data: " + AppContext.getTesting_data().size() + "x" + AppContext.getTesting_data().length()
		+ ", Train #Classes: " + AppContext.getTraining_data().get_num_classes() 
		+ ", Test #Classes: " + AppContext.getTesting_data().get_num_classes());
		System.out.println("Repeats: " + AppContext.num_repeats + " , Trees: " + AppContext.num_trees  
				+ " , Candidates per Split(r): " + AppContext.num_candidates_per_split);
		System.out.println("Output Dir: " + AppContext.output_dir + ", Export: " + AppContext.export_level + ", Verbosity: " + AppContext.verbosity);
		System.out.println("Select DM per node: " + AppContext.random_dm_per_node + " , Shuffle Data: " + AppContext.shuffle_dataset + ", JVM WarmUp: " + AppContext.warmup_java);
		System.out.println("----------------------------------------------------------------------------------------------------");
		 

	}

}
