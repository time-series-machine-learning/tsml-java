package trees;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import core.AppContext;
import core.ProximityForestResult;
import core.contracts.Dataset;
import util.PrintUtilities;
import utilities.Utilities;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class ProximityForest implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -1183368028217094381L;
	protected transient ProximityForestResult result;
	protected int forest_id;
	protected ProximityTree trees[];
	public String prefix;
	
	int[] num_votes;
	List<Integer> max_voted_classes;
	
	public ProximityForest(int forest_id) {

		this.result = new ProximityForestResult(this);
		
		this.forest_id = forest_id;
		this.trees = new ProximityTree[AppContext.num_trees];
		
		for (int i = 0; i < AppContext.num_trees; i++) {
			trees[i] = new ProximityTree(i, this);
		}

	}

	public double[] predict_proba(double[] query, int numClasses) throws Exception {
		num_votes = new int[numClasses];
		max_voted_classes = new ArrayList<>();

		int label;
		int max_vote_count = -1;
		int temp_count = 0;

		for (int i = 0; i < num_votes.length; i++) {
			num_votes[i] = 0;
		}
		max_voted_classes.clear();

		for (int i = 0; i < trees.length; i++) {
			label = trees[i].predict(query);

			num_votes[label]++;
		}

		return Utilities.normalise(num_votes);
	}
	
	public void train(Dataset train_data) throws Exception {
		result.startTimeTrain = System.nanoTime();


		System.setOut(new PrintStream(new BufferedOutputStream(new FileOutputStream("their_pf.out")), true));
		System.setErr(new PrintStream(new BufferedOutputStream(new FileOutputStream("their_pf.err")), true));

		for (int i = 0; i < AppContext.num_trees; i++) {
			trees[i].train(train_data);
			
			if (AppContext.verbosity > 0) {
				System.out.print(i+".");
				if (AppContext.verbosity > 1) {
					PrintUtilities.printMemoryUsage(true);	
					if ((i+1) % 20 == 0) {
						System.out.println();
					}
				}		
			}

		}
		
		result.endTimeTrain = System.nanoTime();
		result.elapsedTimeTrain = result.endTimeTrain - result.startTimeTrain;
		
		if (AppContext.verbosity > 0) {
			System.out.print("\n");				
		}
		
//		System.gc();
		if (AppContext.verbosity > 0) {
			PrintUtilities.printMemoryUsage();	
		}
	
	}
	
	//ASSUMES CLASS labels HAVE BEEN reordered to start from 0 and contiguous
	public ProximityForestResult test(Dataset test_data) throws Exception {
		result.startTimeTest = System.nanoTime();
		
		num_votes = new int[test_data._get_initial_class_labels().size()];
		max_voted_classes = new ArrayList<Integer>();		
		
		int predicted_class;
		int actual_class;
		int size = test_data.size();
		
		for (int i=0; i < size; i++){
			actual_class = test_data.get_class(i);
			predicted_class = predict(test_data.get_series(i));
			if (actual_class != predicted_class){
				result.errors++;
			}else{
				result.correct++;
			}
			
			if (AppContext.verbosity > 0) {
				if (i % AppContext.print_test_progress_for_each_instances == 0) {
					System.out.print("*");
				}				
			}
		}
		
		result.endTimeTest = System.nanoTime();
		result.elapsedTimeTest = result.endTimeTest - result.startTimeTest;
		
		if (AppContext.verbosity > 0) {
			System.out.println();
		}
		
		
		assert test_data.size() == result.errors + result.correct;		
		result.accuracy  = ((double) result.correct) / test_data.size();
		result.error_rate = 1 - result.accuracy;

        return result;
	}
	
	public Integer predict(double[] query) throws Exception {
		//ASSUMES CLASSES HAVE BEEN REMAPPED, start from 0
		int label;
		int max_vote_count = -1;
		int temp_count = 0;
		
		for (int i = 0; i < num_votes.length; i++) {
			num_votes[i] = 0;
		}
		max_voted_classes.clear();

		for (int i = 0; i < trees.length; i++) {
			label = trees[i].predict(query);
			
			num_votes[label]++;
		}
		
//			System.out.println("vote counting using uni dist");
			
		for (int i = 0; i < num_votes.length; i++) {
			temp_count = num_votes[i];
			
			if (temp_count > max_vote_count) {
				max_vote_count = temp_count;
				max_voted_classes.clear();
				max_voted_classes.add(i);
			}else if (temp_count == max_vote_count) {
				max_voted_classes.add(i);
			}
		}
		
		int r = AppContext.getRand().nextInt(max_voted_classes.size());
		
		//collecting some stats
		if (max_voted_classes.size() > 1) {
			this.result.majority_vote_match_count++;
		}
		
		return max_voted_classes.get(r);
	}
	
	public ProximityTree[] getTrees() {
		return this.trees;
	}
	
	public ProximityTree getTree(int i) {
		return this.trees[i];
	}

	public ProximityForestResult getResultSet() {
		return result;
	}

	public ProximityForestResult getForestStatCollection() {
		
		result.collateResults();
		
		return result;
	}

	public int getForestID() {
		return forest_id;
	}

	public void setForestID(int forest_id) {
		this.forest_id = forest_id;
	}




	
}
