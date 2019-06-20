package classifiers.distance_based.elastic_ensemble.iteration.linear;

import java.util.Collection;
import java.util.List;

public class LinearIterator<A> extends AbstractLinearIterator<A, LinearIterator<A>> {

    public LinearIterator(final List<A> values) {
        super(values);
    }

    public LinearIterator() {
        super();
    }

    public LinearIterator(LinearIterator<A> other) {
        this(other.values);
        index = other.index;
    }

    @Override
    public LinearIterator<A> iterator() {
        return new LinearIterator<>(this);
    }
}
