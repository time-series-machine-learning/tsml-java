package tsml.classifiers.distance_based.proximity.tree;
/*

purpose: // todo - docs - type the purpose of the code here

created edited by goastler on 17/02/2020
    
*/

import utilities.Utilities;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

public class BaseTree<A> {

    public BaseTree() {

    }

    private TreeNode<? extends A> root = null;

    public TreeNode<? extends A> getRoot() {
        return root;
    }

    public void setRoot(TreeNode<? extends A> root) {
        this.root = root;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    public int size() {
        List<TreeNode<? extends A>> backlog = new LinkedList<>();
        ListIterator<TreeNode<? extends A>> iterator = backlog.listIterator();
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
