package tsml.classifiers.distance_based.proximity.tree;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;

import utilities.Utilities;

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
        setElement(element);
    }

    @Override
    public TreeNode<A> getParent() {
        return parent;
    }

    @Override
    public void setParent(TreeNode<A> parent) {
        if(parent != null && !this.parent.equals(parent)) {
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
        List<TreeNode<?>> backlog = new LinkedList<>();
        ListIterator<TreeNode<?>> iterator = backlog.listIterator();
        iterator.add(this);
        return Utilities.sum(iterator, node -> {
            node.getChildren().forEach(iterator::add);
            return node.getChildren().size();
        }) + 1; // + 1 to count this node
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
        int height = 0;
        int maxHeight = 0;
        TreeNode<?> node = this;
        LinkedList<TreeNode<?>> nodeStack = new LinkedList<>();
        LinkedList<Iterator<? extends TreeNode<?>>> childrenIndexStack = new LinkedList<>();
        while (true) {
            height++;
            if(!node.isLeaf()) {
                nodeStack.add(node);
                Iterator<? extends TreeNode<?>> iterator = node.getChildren().iterator();
                node = iterator.next();
                childrenIndexStack.add(iterator);
            } else {
                // node is leaf, therefore need to check max height
                maxHeight = Math.max(maxHeight, height);
                if(nodeStack.isEmpty()) {
                    break;
                }
                // then we need to go up 1 level to the parent
                height--;
                TreeNode<?> parent = nodeStack.pop();
                // get the current child we've been looking at
                Iterator<? extends TreeNode<?>> index = childrenIndexStack.pop();
                // if there's remaining children to be examined
                if(index.hasNext()) {
                    // assign current node the next child
                    node = index.next();
                    // add the parent back onto the stack
                    nodeStack.add(parent);
                    // add the index of the current child onto the stack
                    childrenIndexStack.add(index);
                } else {
                    // we've examined all the children
                    // therefore assign current node the parent
                    node = parent;
                    // and loop again to check the parent isn't a leaf, etc, etc
                }
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
}
