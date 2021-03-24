package tsml.classifiers.distance_based.utils.collections.iteration;

public class EmptyIterator<A> implements DefaultListIterator<A> {
    @Override public boolean hasNext() {
        return false;
    }

    @Override public boolean hasPrevious() {
        return false;
    }
}
