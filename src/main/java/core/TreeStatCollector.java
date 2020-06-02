package core;

import trees.ProximityTree;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class TreeStatCollector {

	private transient ProximityTree tree;
	
	public int forest_id;
	public int tree_id;
	
	//tree stats
	public int num_nodes;
	public int num_leaves;
	public int depth;
	public double weighted_depth;
	
	
	public TreeStatCollector(int forest_id, int tree_id){		
		this.forest_id = forest_id;
		this.tree_id = tree_id;
	}
	
	public void collateResults(ProximityTree tree) {
		this.tree = tree;
		
		num_nodes = tree.get_num_nodes();
		num_leaves = tree.get_num_leaves();
		depth = tree.get_height();
		weighted_depth = -1; //TODO not implemented yet tree.get_weighted_depth();
	}
	
}
