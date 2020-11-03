package tsml.classifiers.distance_based.utils.collections.tree;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface Tree<A> extends Serializable, List<TreeNode<A>> {

    TreeNode<A> getRoot();

    void setRoot(TreeNode<A> root);

    int size();

    int height();

    void clear();
}
