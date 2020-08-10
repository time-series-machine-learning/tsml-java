package tsml.classifiers.distance_based.utils.collections.iteration;

import java.util.List;

/**
 * Purpose: round robin traverse a list.
 *
 * Contributors: goastler
 *
 * @param <A>
 */
public class RoundRobinListIterator<A> extends LinearListIterator<A> {
    public RoundRobinListIterator(List<A> list) {
        super(list);
    }

    public RoundRobinListIterator() {}

    @Override
    public A next() {
        A next = super.next();
        if(index == list.size()) {
            index = 0;
        }
        return next;
    }

    @Override
    public void remove() {
        super.remove();
        if(index < 0) {
            index = list.size() - 1;
        }
    }

    @Override
    public int nextIndex() {
        return super.nextIndex() % list.size();
    }
}
