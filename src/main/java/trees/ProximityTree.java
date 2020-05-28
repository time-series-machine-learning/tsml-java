package trees;

import java.lang.reflect.Field;
import java.util.Random;

import core.AppContext;
import core.TreeStatCollector;
import core.contracts.Dataset;
import distance.elastic.DistanceMeasure;
import tsml.classifiers.distance_based.utils.classifier_mixins.Copy;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class ProximityTree{
	protected int forest_id;	
	private int tree_id;
	protected Node root;
	protected int node_counter = 0;
	
	protected transient Random rand;
	public TreeStatCollector stats;
	
	protected DistanceMeasure tree_distance_measure; //only used if AppContext.random_dm_per_node == false

	public ProximityTree(int tree_id, ProximityForest forest) {
		this.forest_id = forest.forest_id;
		this.tree_id = tree_id;
		this.rand = AppContext.getRand();
		stats = new TreeStatCollector(forest_id, tree_id);
	}

	public Node getRootNode() {
		return this.root;
	}
	
	public void train(Dataset data) throws Exception {
		
		
		if (AppContext.random_dm_per_node ==  false) {	//DM is selected once per tree
			int r = AppContext.getRand().nextInt(AppContext.enabled_distance_measures.length);
			tree_distance_measure = new DistanceMeasure(AppContext.enabled_distance_measures[r]);		
			//params selected per node in the splitter class
		}
		
		this.root = new Node(null, null, ++node_counter, this);
		this.root.train(data);
	}
	
	public Integer predict(double[] query) throws Exception {
		Node node = this.root;

		while(!node.is_leaf()) {
			node = node.children[node.splitter.find_closest_branch(query)];
		}

		return node.label();
	}	

	
	public int getTreeID() {
		return tree_id;
	}

	
	//************************************** START stats -- development/debug code
	public TreeStatCollector getTreeStatCollection() {
		
		stats.collateResults(this);
		
		return stats;
	}	
	
	public int get_num_nodes() {
		if (node_counter != get_num_nodes(root)) {
			System.out.println("Error: error in node counter!");
			return -1;
		}else {
			return node_counter;
		}
	}	

	public int get_num_nodes(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 1;
		}
		
		for (int i = 0; i < n.children.length; i++) {
			count+= get_num_nodes(n.children[i]);
		}
		
		return count+1;
	}
	
	public int get_num_leaves() {
		return get_num_leaves(root);
	}	
	
	public int get_num_leaves(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 1;
		}
		
		for (int i = 0; i < n.children.length; i++) {
			count+= get_num_leaves(n.children[i]);
		}
		
		return count;
	}
	
	public int get_num_internal_nodes() {
		return get_num_internal_nodes(root);
	}
	
	public int get_num_internal_nodes(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 0;
		}
		
		for (int i = 0; i < n.children.length; i++) {
			count+= get_num_internal_nodes(n.children[i]);
		}
		
		return count+1;
	}
	
	public int get_height() {
		return get_height(root);
	}
	
	public int get_height(Node n) {
		int max_depth = 0;
		
		if (n.children == null) {
			return 0;
		}

		for (int i = 0; i < n.children.length; i++) {
			max_depth = Math.max(max_depth, get_height(n.children[i]));
		}
		
		return max_depth+1;
	}
	
	public int get_min_depth(Node n) {
		int max_depth = 0;
		
		if (n.children == null) {
			return 0;
		}

		for (int i = 0; i < n.children.length; i++) {
			max_depth = Math.min(max_depth, get_height(n.children[i]));
		}
		
		return max_depth+1;
	}
	
//	public double get_weighted_depth() {
//		return printTreeComplexity(root, 0, root.data.size());
//	}
//	
//	// high deep and unbalanced
//	// low is shallow and balanced?
//	public double printTreeComplexity(Node n, int depth, int root_size) {
//		double ratio = 0;
//		
//		if (n.is_leaf) {
//			double r = (double)n.data.size()/root_size * (double)depth;
////			System.out.format("%d: %d/%d*%d/%d + %f + ", n.label, 
////					n.data.size(),root_size, depth, max_depth, r);
//			
//			return r;
//		}
//		
//		for (int i = 0; i < n.children.length; i++) {
//			ratio += printTreeComplexity(n.children[i], depth+1, root_size);
//		}
//		
//		return ratio;
//	}		
	
	
	//**************************** END stats -- development/debug code
	
	
	
	
	
	
	
	public class Node{
	
		protected transient Node parent;	//dont need this, but it helps to debug
		protected transient ProximityTree tree;		
		
		protected int node_id;
		protected int node_depth = 0;

		protected boolean is_leaf = false;
		protected Integer label;

//		protected transient Dataset data;				
		protected Node[] children;
		protected Splitter splitter;
		
		public Node(Node parent, Integer label, int node_id, ProximityTree tree) {
			this.parent = parent;
//			this.data = new ListDataset();
			this.node_id = node_id;
			this.tree = tree;
			
			if (parent != null) {
				node_depth = parent.node_depth + 1;
			}
		}
		
		public boolean is_leaf() {
			return this.is_leaf;
		}
		
		public Integer label() {
			return this.label;
		}	
		
		public Node[] get_children() {
			return this.children;
		}		
		
//		public Dataset get_data() {
//			return this.data;
//		}		
		
		public String toString() {
			return "d: ";// + this.data.toString();
		}		

		
//		public void train(Dataset data) throws Exception {
//			this.data = data;
//			this.train();
//		}		
		
		public void train(Dataset data) throws Exception {
//			System.out.println(this.node_depth + ":   " + (this.parent == null ? "r" : this.parent.node_id)  +"->"+ this.node_id +":"+ data.toString());
			
			//Debugging check
			if (data == null || data.size() == 0) {
				throw new Exception("possible bug: empty node found");
//				this.is_leaf = true;
//				return;
			}
			
			if (data.gini() == 0) {
				this.label = data.get_class(0);
				this.is_leaf = true;
				return;
			}

			this.splitter = new Splitter(this);
						
			Dataset[] best_splits = splitter.find_best_split(data);
			String str = splitter.distance_measure.toString();
			String a = splitter.distance_measure.toString();
			if(a.endsWith("cv")) {
				a = a.substring(0, a.length() - 2);
			}
			for(Field field : Copy.findFields(splitter.distance_measure.getClass())) {
				field.setAccessible(true);
				if(field.getName().endsWith(a.toUpperCase()) && field.getName().length() > a.length()) {
					str += ", " + field.getName() + "=" + field.get(splitter.distance_measure);
				}
			}
			System.out.println(str);
			System.out.println(splitter.best_weighted_gini);
			this.children = new Node[best_splits.length];
			for (int i = 0; i < children.length; i++) {
				this.children[i] = new Node(this, i, ++tree.node_counter, tree);
			}

			for (int i = 0; i < best_splits.length; i++) {

				this.children[i].train(best_splits[i]);
			}
		}

	}
	
}
