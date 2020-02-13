package tsml.classifiers.distance_based.interval;

import java.util.ArrayList;
import java.util.List;

public class Node {
    private List<Node> children = new ArrayList<>();

    public List<Node> getChildren() {
        return children;
    }

    public void setChildren(final List<Node> children) {
        this.children = children;
    }
}
