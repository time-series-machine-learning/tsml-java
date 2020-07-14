package tsml.classifiers.distance_based.utils.collections.tree;

import java.util.Collection;
import java.util.Iterator;
import java.util.function.Predicate;

/**
 * Purpose: a tree data structure.
 *
 * Contributors: goastler
 */

public class BaseTree<A> implements Tree<A> {

    public BaseTree() {

    }

    @Override
    public void clear() {
        setRoot(null);
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

    @Override public boolean isEmpty() {
        return root.isEmpty();
    }

    @Override public boolean contains(final Object o) {
        return root.contains(o);
    }

    @Override public Iterator<TreeNode<A>> iterator() {
        return root.iterator();
    }

    @Override public Object[] toArray() {
        return root.toArray();
    }

    @Override public <T> T[] toArray(final T[] ts) {
        return root.toArray(ts);
    }

    @Override public boolean add(final TreeNode<A> aTreeNode) {
        return root.add(aTreeNode);
    }

    @Override public boolean remove(final Object o) {
        return root.remove(o);
    }

    @Override public boolean containsAll(final Collection<?> collection) {
        return root.containsAll(collection);
    }

    @Override public boolean addAll(
            final Collection<? extends TreeNode<A>> collection) {
        return root.addAll(collection);
    }

    @Override public boolean removeAll(final Collection<?> collection) {
        return root.removeAll(collection);
    }

    @Override public boolean retainAll(final Collection<?> collection) {
        return root.retainAll(collection);
    }
}
