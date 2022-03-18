package tsml.classifiers.distance_based.utils.collections.iteration;

import java.util.Iterator;
import java.util.Objects;

public class FlatIterator<A> implements Iterator<A> {
    
    public FlatIterator(Iterable<? extends Iterable<A>> iterables) {
        mainIterator = new TransformIterator<>(iterables, Iterable::iterator);
    }

    private final Iterator<? extends Iterator<A>> mainIterator;
    private Iterator<A> subIterator;
    
    @Override public boolean hasNext() {
        while((subIterator == null || !subIterator.hasNext()) && mainIterator.hasNext()) {
            subIterator = mainIterator.next();
        }
        // no iterators in main therefore sub iterator null
        if(subIterator == null) {
            return false;
        }
        // otherwise sub iterator is a viable iterator
        return subIterator.hasNext();
    }

    @Override public A next() {
        return subIterator.next();
    }
}
