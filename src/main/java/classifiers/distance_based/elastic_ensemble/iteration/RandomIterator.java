package classifiers.distance_based.elastic_ensemble.iteration;

import java.util.List;
import java.util.Random;

public class RandomIterator<A> extends LinearIterator<A> {
    private final Random random;

    public RandomIterator(final List<? extends A> values, final Random random) {
        super(values);
        this.random = random;
    }

    @Override
    public void remove() {
        values.remove(index--);
    }


    @Override
    public A next() {
        return values.get(index = random.nextInt(values.size()));
    }
}
