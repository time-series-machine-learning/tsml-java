package tsml.classifiers.distance_based.utils.collections.tree;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;

/**
 * Purpose: a node of a tree. Tree nodes are a list of their child nodes.
 * <p>
 * Contributors: goastler
 */

public interface TreeNode<A> extends Serializable, List<TreeNode<A>> {

    TreeNode<A> getParent();

    void setParent(TreeNode<A> parent);
    
    List<TreeNode<A>> getChildren();
    
    void setChildren(List<TreeNode<A>> children);

    A getValue();

    void setValue(A element);

    boolean hasValue();

    int numChildren();

    int size();

    boolean isLeaf();

    int height();

    int getLevel();

    boolean isRoot();

    @Override TreeNode<A> get(int i);

    @Override TreeNode<A> set(int i, TreeNode<A> aTreeNode);

    @Override TreeNode<A> remove(int i);

    @Override boolean remove(Object o);

    @Override void clear();
}
