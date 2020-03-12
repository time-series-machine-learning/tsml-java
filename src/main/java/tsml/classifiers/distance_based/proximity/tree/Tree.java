package tsml.classifiers.distance_based.proximity.tree;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface Tree<A> {

    TreeNode<? extends A> getRoot();

    void setRoot(TreeNode<? extends A> root);

    int size();

    int height();
}
