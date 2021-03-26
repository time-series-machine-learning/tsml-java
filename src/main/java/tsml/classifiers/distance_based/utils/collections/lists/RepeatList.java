package tsml.classifiers.distance_based.utils.collections.lists;

import java.util.AbstractList;

public class RepeatList<A> extends AbstractList<A> {

    public RepeatList(final A element, final int size) {
        this.element = element;
        this.size = size;
    }

    private final A element;
    private final int size;
    
    @Override public A get(final int i) {
        return element;
    }

    @Override public int size() {
        return size;
    }
}
