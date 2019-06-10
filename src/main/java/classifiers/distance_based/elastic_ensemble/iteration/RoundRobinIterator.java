package classifiers.distance_based.elastic_ensemble.iteration;

import java.util.List;

public class RoundRobinIterator<A> extends LinearIterator<A>
     {

    public RoundRobinIterator(final List<? extends A> values) {
        super(values);
    }

    @Override
    public void remove() {
        values.remove(index--);
    }

    @Override
    public A next() {
        return values.get(index = (index + 1) % values.size());
    }
}
