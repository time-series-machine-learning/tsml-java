package tsml.classifiers.distance_based.utils.collections.tree;

import utilities.Utilities;

import java.io.Serializable;
import java.util.*;

/**
 * Purpose: a node of a tree. Tree nodes are a list of their child nodes.
 * <p>
 * Contributors: goastler
 */

public interface TreeNode<A> extends Serializable, List<TreeNode<A>> {

    TreeNode<A> getParent();

    void setParent(TreeNode<A> parent);
    
    A getValue();

    void setValue(A element);
    
    default boolean isLeaf() {
        return isEmpty();
    }
    
    default int numChildren() {
        return size();
    }
    
    default int numNodes() {
        // +1 for current node
        return 1 + Utilities.apply(this, TreeNode::numNodes).stream().mapToInt(i -> i).sum();
    }

    default int height() {
        int height = 1; // start at 1 because this node is 1 level itself
        int maxHeight = 1;
        LinkedList<TreeNode<?>> nodeStack = new LinkedList<>();
        LinkedList<Iterator<? extends TreeNode<?>>> childrenIteratorStack = new LinkedList<>();
        nodeStack.add(this);
        childrenIteratorStack.add(this.iterator());
        while (!nodeStack.isEmpty()) {
            TreeNode<?> node = nodeStack.peekLast();
            Iterator<? extends TreeNode<?>> iterator = childrenIteratorStack.peekLast();
            if(iterator.hasNext()) {
                // descend down to next child
                height++;
                node = iterator.next();
                iterator = node.iterator();
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

    default int level() {
        TreeNode<A> parent = getParent();
        int level = 0;
        while(parent != null) {
            parent = parent.getParent();
            level++;
        }
        return level;
    }

    default boolean isRoot() {
        return getParent() == null;
    }

    @Override TreeNode<A> get(int i);

    @Override TreeNode<A> set(int i, TreeNode<A> child);

    @Override TreeNode<A> remove(int i);

    @Override void clear();
    
    default void removeParent() {
        setParent(null);
    }
    
    /**
     * Get the location of the node within the tree. This is separated by dots on each level
     * @return
     */
    default List<Integer> getLocation() {
        TreeNode<A> parent = getParent();
        final List<Integer> location;
        if(parent == null) {
            location = new ArrayList<>();
        } else {
            location = parent.getLocation();
            location.add(getParent().indexOf(this));
        }
        return location;
    }
}
