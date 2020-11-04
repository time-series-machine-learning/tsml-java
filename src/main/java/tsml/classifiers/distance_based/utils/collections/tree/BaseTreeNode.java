package tsml.classifiers.distance_based.utils.collections.tree;

import java.util.*;

/**
 * Purpose: node of a tree data structure.
 *
 * Contributors: goastler
 */
public class BaseTreeNode<A> extends AbstractList<TreeNode<A>> implements TreeNode<A> {

    private final List<TreeNode<A>> children = new ArrayList<>();
    private A element;
    private TreeNode<A> parent;
    
    public BaseTreeNode() {}

    public BaseTreeNode(A element) {
        this(element, null);
    }

    public BaseTreeNode(A element, TreeNode<A> parent) {
        setValue(element);
        setParent(parent);
    }

    @Override public boolean equals(final Object o) {
        return this == o;
    }

    @Override
    public TreeNode<A> getParent() {
        return parent;
    }

    @Override
    public void setParent(TreeNode<A> nextParent) {
        if(parent == nextParent) {
            // do nothing
            return;
        }
        if(parent != null) {
            // remove this child from the parent
            parent.remove(this);
        }
        if(nextParent != null) {
            if(!nextParent.contains(this)) {
                // add this child to the new parent
                nextParent.add(this);
            }
        }
        this.parent = nextParent;
    }

    @Override
    public A getValue() {
        return element;
    }

    @Override
    public void setValue(A element) {
        this.element = element;
    }

    /**
     * total number of nodes in the tree, including this one
     * @return
     */
    @Override
    public int size() {
        return children.size();
    }

    @Override public void add(final int i, final TreeNode<A> node) {
        if(node.getParent() == this) {
            throw new IllegalArgumentException("already a child");
        }
        // node is not a child yet
        // add the node to the children
        children.add(i, node);
        // set this node as the parent
        node.setParent(this);
    }

    @Override public boolean add(final TreeNode<A> node) {
        int size = children.size();
        add(children.size(), node);
        return size != children.size();
    }

    @Override public TreeNode<A> get(final int i) {
        return children.get(i);
    }

    @Override public TreeNode<A> set(final int i, final TreeNode<A> child) {
        if(child.getParent() == this) {
            // already a child - cannot house multiple children
            throw new IllegalArgumentException("already a child: " + child);
        }
        // get the previous
        TreeNode<A> previous = children.get(i);
        // overwrite the previous
        children.set(i, child);
        // remove this as the parent of the overwritten
        previous.removeParent();
        // setup the new node as a child
        child.setParent(this);
        return previous;
    }

    @Override public TreeNode<A> remove(final int i) {
        // remove the child
        TreeNode<A> child = children.remove(i);
        // discard the parent
        child.removeParent();
        return child;
    }

    @Override public String toString() {
        return "BaseTreeNode{" +
                       "location=" + getLocation() +
                       ", element=" + element +
                       ", children=" + children +
                       '}';
    }

    @Override public void clear() {
        // remove all the children
        for(int i = children.size() - 1; i >= 0; i--) {
            final TreeNode<A> child = children.get(i);
            child.removeParent();
        }
    }
}
