package tsml.classifiers.distance_based.utils.collections.iteration;

import java.util.List;

/**
 * Purpose: linearly traverse a list.
 *
 * Contributors: goastler
 *
 * @param <A>
 */
public class LinearIterator<A> extends AbstractListIterator<A> {

    public LinearIterator(final List<A> list) {
        super(list);
    }

    public LinearIterator() {

    }

    @Override protected int findNextIndex() {
        return getIndex() + 1;
    }

}
