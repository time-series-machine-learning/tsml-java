package tsml.classifiers.distance_based.proximity.tree;

/**
 * Purpose: a tree data structure.
 *
 * Contributors: goastler
 */

import utilities.Utilities;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

public class Tree<A> {

    public Tree() {

    }

    private TreeNode<? extends A> root = null;

    public TreeNode<? extends A> getRoot() {
        return root;
    }

    public void setRoot(TreeNode<? extends A> root) {
        this.root = root;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    /**
     * total number of nodes in the tree
     * @return
     */
    public int size() {
        if(root == null) {
            return 0;
        } else {
            return root.size();
        }
    }

    public int height() {
        if(root == null) {
            return 0;
        } else {
            return root.height();
        }
    }
}
