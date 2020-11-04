package tsml.classifiers.distance_based.utils.collections.tree;

import java.io.Serializable;
import java.util.List;

/**
 *
 *  Purpose: a tree data structure. Imposes no ideology regarding context of nodes, e.g. sorted / balanced, etc. 
 *  This solely provide structural implementation and leave the responsibility of maintaining a coherent context to 
 *  an implementing class.
 * <p>
 * Contributors: goastler
 */

public interface Tree<A> extends Serializable, List<A> {

    TreeNode<A> getRoot();

    void setRoot(TreeNode<A> root);

    /**
     * Return the number of nodes in the tree
     * @return
     */
    default int size() {
        TreeNode<A> root = getRoot();
        if(root == null) {
            return 0;
        }
        return root.size();
    }
    
    /**
     * Return the max depth. This is measured with root node == height 0.
     * @return
     */
    default int height() {
        TreeNode<A> root = getRoot();
        if(root == null) {
            return 0;
        }
        return root.height();
    }

    default void clear() {
        setRoot(null);
    }

    default int nodeCount() {
        TreeNode<A> root = getRoot();
        if(root == null) {
            return 0;
        }
        return root.numNodes();
    }
    
}
