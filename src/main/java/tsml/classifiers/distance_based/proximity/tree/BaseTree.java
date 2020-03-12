package tsml.classifiers.distance_based.proximity.tree;

/**
 * Purpose: a tree data structure.
 *
 * Contributors: goastler
 */

public class BaseTree<A> implements Tree<A> {

    public BaseTree() {

    }

    private TreeNode<A> root = null;

    @Override
    public TreeNode<A> getRoot() {
        return root;
    }

    @Override
    public void setRoot(TreeNode<A> root) {
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
    @Override
    public int size() {
        if(root == null) {
            return 0;
        } else {
            return root.size();
        }
    }

    @Override
    public int height() {
        if(root == null) {
            return 0;
        } else {
            return root.height();
        }
    }
}
