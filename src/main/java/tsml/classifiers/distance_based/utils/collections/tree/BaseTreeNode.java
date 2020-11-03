package tsml.classifiers.distance_based.utils.collections.tree;

import java.util.*;

/**
 * Purpose: node of a tree data structure.
 *
 * Contributors: goastler
 */
public class BaseTreeNode<A> implements TreeNode<A> {

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

    @Override
    public TreeNode<A> getParent() {
        return parent;
    }

    @Override
    public void setParent(TreeNode<A> parent) {
        if(parent != null && (this.parent == null || !this.parent.equals(parent))) {
            parent.getChildren().add(this);
        }
        this.parent = parent;
    }

    @Override
    public A getValue() {
        return element;
    }

    @Override
    public void setValue(A element) {
        this.element = element;
    }

    @Override
    public boolean hasValue() {
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
        TreeNode<A> parent = this.parent;
        int level = 0;
        while(parent != null) {
            parent = parent.getParent();
            level++;
        }
        return level;
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

    @Override public String toString() {
        return "BaseTreeNode{" +
                       "location=" + getLocation() +
                       "element=" + element +
                       "children=" + children +
                       '}';
    }

    /**
     * Get the location of the node within the tree. This is separated by dots on each level
     * @return
     */
    public String getLocation() {
        
    }
}
