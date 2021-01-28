package tsml.classifiers.distance_based.utils.collections.lists;

import java.util.AbstractList;

public class IndexList extends AbstractList<Integer> {

    public IndexList(final int size) {
        if(size < 0) {
            throw new IllegalArgumentException("size cannot be less than 0");
        }
        this.size = size;
    }

    private final int size;
    
    @Override public Integer get(final int i) {
        return i;
    }

    @Override public int size() {
        return size;
    }
}
