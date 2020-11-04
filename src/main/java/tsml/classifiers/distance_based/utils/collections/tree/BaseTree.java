package tsml.classifiers.distance_based.utils.collections.tree;

import java.util.AbstractList;
import java.util.Collection;
import java.util.Iterator;

/**
 * A general tree data structure. No context of nodes at all, just a simple hierarchy.
 *
 * Contributors: goastler
 */

public class BaseTree<A> extends AbstractList<A> implements Tree<A> {

    public BaseTree() {
        clear();
    }

    private TreeNode<A> root;

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
        return getClass().getSimpleName() + "{root=" + root + "}";
    }

    public A get(int i) {
        throw new UnsupportedOperationException("get not implemented for tree");
    }

    @Override public int size() {
        return Tree.super.size();
    }

    @Override public boolean equals(final Object o) {
        return this == o;
    }
}
