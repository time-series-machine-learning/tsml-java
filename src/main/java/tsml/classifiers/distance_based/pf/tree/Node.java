package tsml.classifiers.distance_based.pf.tree;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Node<A> {
    private List<Node<? extends A>> children = new ArrayList<>();
    private A element;
    private int level = -1;

    public Node() {}

    public Node(A element) {
        setElement(element);
    }

    public List<Node<? extends A>> getChildren() {
        return children;
    }

    public void setChildren(final List<Node<? extends A>> children) {
        if(children == null) {
            throw new IllegalArgumentException("cannot have null as children");
        }
        this.children = children;
    }

    public A getElement() {
        return element;
    }

    public void setElement(A element) {
        this.element = element;
    }

    public boolean hasElement() {
        return element != null;
    }

    public int size() {
        return children.size();
    }

    public boolean isLeaf() {
        return children.isEmpty();
    }

    public int height() { // todo test this out as first version so may contain bugs
        int height = 0;
        int maxHeight = 0;
        Node<?> node = this;
        LinkedList<Node<?>> nodeStack = new LinkedList<>();
        LinkedList<Integer> childrenIndexStack = new LinkedList<>();
        while (true) {
            height++;
            if(!node.isLeaf()) {
                nodeStack.add(node);
                node = node.getChildren().get(0);
                childrenIndexStack.add(0);
            } else {
                // node is leaf, therefore need to check max height
                maxHeight = Math.max(maxHeight, height);
                if(nodeStack.isEmpty()) {
                    break;
                }
                // then we need to go up 1 level to the parent
                height--;
                Node<?> parent = nodeStack.pop();
                // get the current child we've been looking at
                Integer index = childrenIndexStack.pop();
                int nextIndex = index + 1;
                // if there's remaining children to be examined
                if(nextIndex < parent.getChildren().size()) {
                    // assign current node the next child
                    node = parent.getChildren().get(nextIndex);
                    // add the parent back onto the stack
                    nodeStack.add(parent);
                    // add the index of the current child onto the stack
                    childrenIndexStack.add(nextIndex);
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

    public int getLevel() {
        return level;
    }

    public void setLevel(int level) {
        this.level = level;
    }

    public void addChild(Node<A> child) {
        children.add(child);
        child.setLevel(level + 1);
    }
}
