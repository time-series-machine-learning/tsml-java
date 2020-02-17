package tsml.classifiers.distance_based.pf;
/*

purpose: // todo - docs - type the purpose of the code here

created edited by goastler on 17/02/2020
    
*/

import utilities.ArrayUtilities;
import utilities.Utilities;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.function.Function;

public class Tree<A> {

    public Tree() {

    }

    public Tree(Tree<? extends A> other) {
        setRoot(other.getRoot());
    }

    private Node<? extends A> root = null;

    public Node<? extends A> getRoot() {
        return root;
    }

    public void setRoot(Node<? extends A> root) {
        this.root = root;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    public int size() {
        List<Node<? extends A>> backlog = new LinkedList<>();
        ListIterator<Node<? extends A>> iterator = backlog.listIterator();
        iterator.add(root);
        return Utilities.sum(iterator, node -> {
            node.getChildren().forEach(iterator::add);
            return node.getChildren().size();
        });
    }

    public int height() {
        return root.height();
    }
}
