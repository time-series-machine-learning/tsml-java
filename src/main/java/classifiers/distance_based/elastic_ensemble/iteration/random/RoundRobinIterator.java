package classifiers.distance_based.elastic_ensemble.iteration.random;

import java.util.Collection;
import java.util.Random;

public class RoundRobinIterator<A> extends AbstractRoundRobinIterator<A, RoundRobinIterator<A>> {

    public RoundRobinIterator(final Collection<? extends A> values) {
        super(values);
    }

    public RoundRobinIterator(RoundRobinIterator<A> other) {
        this(other.values);
        index = other.index;
    }

    public RoundRobinIterator() {
        super();
    }

    @Override
    public RoundRobinIterator<A> iterator() {
        return new RoundRobinIterator<>(this);
    }
}
