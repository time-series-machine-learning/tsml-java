package classifiers.distance_based.elastic_ensemble.iteration;

import java.util.Iterator;
import java.util.List;

public class LinearIterator<A> implements Iterator<A> {
    protected final List<? extends A> values;
    protected int index = 0;

    public LinearIterator(final List<? extends A> values) {this.values = values;}

    @Override
    public void remove() {
        values.remove(index--);
    }

    @Override
    public boolean hasNext() {
        return !values.isEmpty();
    }

    @Override
    public A next() {
        return values.get(index++);
    }
}
