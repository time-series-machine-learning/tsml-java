package tsml.classifiers.distance_based.utils.collections.tree;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface TreeNode<A> extends Serializable, Collection<TreeNode<A>> {

    TreeNode<A> getParent();

    void setParent(TreeNode<A> parent);

    List<TreeNode<A>> getChildren();

    default boolean hasChildren() {
        return getChildren().size() > 0;
    }

    A getElement();

    void setElement(A element);

    boolean hasElement();

    int numChildren();

    int size();

    boolean isLeaf();

    int height();

    int getLevel();

    boolean addChild(TreeNode<A> child);

    boolean removeChild(TreeNode<A> child);

    boolean isRoot();
}
