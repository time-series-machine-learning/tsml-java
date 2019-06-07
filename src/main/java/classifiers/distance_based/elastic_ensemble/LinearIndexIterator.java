package classifiers.distance_based.elastic_ensemble;

import java.util.Iterator;

public class LinearIndexIterator implements Iterator<Integer> {

    private int index = 0;
    private final int size;

    public LinearIndexIterator(final int size) {this.size = size;}

    @Override
    public boolean hasNext() {
        return index < size;
    }

    @Override
    public Integer next() {
        return index++;
    }
}
