package tsml.classifiers.distance_based.proximity.tree;

import java.util.Set;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface TreeNode<A> {

    TreeNode<A> getParent();

    void setParent(TreeNode<A> parent);

    Set<TreeNode<A>> getChildren();

    void setChildren(Set<TreeNode<A>> children);

    A getElement();

    void setElement(A element);

    boolean hasElement();

    int size();

    boolean isLeaf();

    int height();

    int getLevel();

    void setLevel(int level);

    boolean addChild(TreeNode<A> child);

    boolean removeChild(TreeNode<A> child);
}
