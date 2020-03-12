package tsml.classifiers.distance_based.proximity.tree;

import com.google.common.collect.ImmutableSet;
import java.util.List;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface TreeNode<A> {

    TreeNode<A> getParent();

    void setParent(TreeNode<A> parent);

    List<TreeNode<A>> getChildren();

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
}
