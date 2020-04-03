package tsml.classifiers.distance_based.utils.iteration;

import java.util.*;

/**
 * Purpose: randomly traverse a list. You can optionally remove as next() is called or not, i.e. without and with
 * replacement respectively.
 *
 * Contributors: goastler
 *
 * @param <A>
 */
public class RandomListIterator<A> extends RandomIterator<A> implements DefaultListIterator<A> {

    public RandomListIterator(final int seed) {
        super(seed);
    }

    public RandomListIterator(final Random random) {
        super(random);
    }

    public RandomListIterator(final int seed, final List<A> list) {
        super(seed, list);
    }

    public RandomListIterator(final Random random, final List<A> list) {
        super(random, list);
    }

    // todo implement other list iterator funcs

    @Override
    public void add(final A item) {
        getList().add(item);
        getIndices().add(getList().size() - 1);
    }

}
