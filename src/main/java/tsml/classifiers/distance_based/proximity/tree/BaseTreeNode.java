package tsml.classifiers.distance_based.proximity.tree;

import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;
import java.util.function.Supplier;

public class BaseTreeNode<A> implements TreeNode<A> {
    private Supplier<Set<TreeNode<A>>> supplier = HashSet::new;
    private Set<TreeNode<A>> children = supplier.get();
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
        if(!this.parent.equals(parent)){
            this.parent = parent;
            parent.addChild(this);
            setLevel(parent.getLevel());
        }
    }

    @Override
    public Set<TreeNode<A>> getChildren() {
        return children;
    }

    @Override
    public void setChildren(Set<TreeNode<A>> children) {
        if(children == null) {
            children = supplier.get();
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
    public int size() {
        return children.size();
    }

    @Override
    public boolean isLeaf() {
        return children.isEmpty();
    }

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

    @Override
    public void setLevel(int level) {
        this.level = level;
    }

    @Override
    public boolean addChild(TreeNode<A> child) {
        boolean result = children.add(child);
        if(result) {
            child.setParent(this);
        }
        return result;
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
