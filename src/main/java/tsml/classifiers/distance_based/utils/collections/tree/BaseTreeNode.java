package tsml.classifiers.distance_based.utils.collections.tree;

import java.util.*;

import org.junit.Assert;
import org.junit.Test;

/**
 * Purpose: node of a tree data structure.
 *
 * Contributors: goastler
 */
public class BaseTreeNode<A> implements TreeNode<A> {

    private List<TreeNode<A>> children = new ArrayList<>();
    private A element;
    private int level = 0;
    private TreeNode<A> parent;

    public BaseTreeNode() {}

    public BaseTreeNode(A element) {
        this(element, null);
    }

    public BaseTreeNode(A element, TreeNode<A> parent) {
        setElement(element);
        setParent(parent);
    }

    public BaseTreeNode(TreeNode<A> parent) {
        this(null, parent);
    }

    @Override
    public TreeNode<A> getParent() {
        return parent;
    }

    @Override
    public void setParent(TreeNode<A> parent) {
        if(parent != null && (this.parent == null || !this.parent.equals(parent))) {
            parent.getChildren().add(this);
            setLevel(parent.getLevel() + 1);
        }
        this.parent = parent;
    }

    @Override
    public List<TreeNode<A>> getChildren() {
        return children;
    }

    protected void setChildren(List<TreeNode<A>> children) {
        if(children == null) {
            children = new ArrayList<>();
        }
        this.children = children;
    }

    @Override
    public A getElement() {
        return element;
    }

    @Override
    public void setElement(A element) {
        this.element = element;
    }

    @Override
    public boolean hasElement() {
        return element != null;
    }

    @Override
    public boolean isRoot() {
        return parent == null;
    }

    /**
     * the number of children branching from this node
     * @return
     */
    @Override
    public int numChildren() {
        return children.size();
    }

    /**
     * total number of nodes in the tree, including this one
     * @return
     */
    @Override
    public int size() {
        LinkedList<TreeNode<?>> backlog = new LinkedList<>();
        int count = 0;
        backlog.add(this);
        while(!backlog.isEmpty()) {
            TreeNode<?> node = backlog.pollFirst();
            count++;
            backlog.addAll(node.getChildren());
        }
        return count;
    }

    @Override public boolean isEmpty() {
        return numChildren() == 0;
    }

    @Override public boolean contains(final Object o) {
        for(TreeNode<A> child : children) {
            if(child.contains(o)) {
                return true;
            }
        }
        return false;
    }

    @Override public Iterator<TreeNode<A>> iterator() {
        return children.iterator();
    }

    @Override public Object[] toArray() {
        return children.toArray();
    }

    @Override public <T> T[] toArray(final T[] ts) {
        return children.toArray(ts);
    }

    @Override public boolean add(final TreeNode<A> aTreeNode) {
        return children.add(aTreeNode);
    }

    @Override public boolean remove(final Object o) {
        return children.remove(o);
    }

    @Override public boolean containsAll(final Collection<?> collection) {
        return children.containsAll(collection);
    }

    @Override public boolean addAll(final Collection<? extends TreeNode<A>> collection) {
        return children.addAll(collection);
    }

    @Override public boolean removeAll(final Collection<?> collection) {
        return children.removeAll(collection);
    }

    @Override public boolean retainAll(final Collection<?> collection) {
        return children.retainAll(collection);
    }

    @Override public void clear() {
        children.clear();
    }

    @Override
    public boolean isLeaf() {
        return children.isEmpty();
    }

    /**
     * the height downwards from this node
     * @return
     */
    @Override
    public int height() { // todo test this out as first version so may contain bugs
        int height = 1; // start at 1 because this node is 1 level itself
        int maxHeight = 1;
        LinkedList<TreeNode<?>> nodeStack = new LinkedList<>();
        LinkedList<Iterator<? extends TreeNode<?>>> childrenIteratorStack = new LinkedList<>();
        nodeStack.add(this);
        childrenIteratorStack.add(this.getChildren().iterator());
        while (!nodeStack.isEmpty()) {
            TreeNode<?> node = nodeStack.peekLast();
            Iterator<? extends TreeNode<?>> iterator = childrenIteratorStack.peekLast();
            if(iterator.hasNext()) {
                // descend down to next child
                height++;
                node = iterator.next();
                iterator = node.getChildren().iterator();
                nodeStack.add(node);
                childrenIteratorStack.add(iterator);
            } else {
                // ascend up to parent
                maxHeight = Math.max(height, maxHeight);
                height--;
                nodeStack.pollLast();
                childrenIteratorStack.pollLast();
            }
        }
        return maxHeight;
    }

    @Override
    public int getLevel() {
        return level;
    }

    protected void setLevel(int level) {
        this.level = level;
    }

    @Override
    public boolean addChild(TreeNode<A> child) {
        child.setParent(this);
        return true;
    }

    @Override
    public boolean removeChild(TreeNode<A> child) {
        boolean result = children.remove(child);
        if(result) {
            if(child.getParent().equals(this)) {
                child.setParent(null);
            }
        }
        return result;
    }

    @Override
    public String toString() {
        return element.toString();
    }
}
