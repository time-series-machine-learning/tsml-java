package tsml.classifiers.distance_based.utils.collections.iteration;

import java.io.Serializable;
import java.util.Iterator;

public class TransformIterator<A, B> implements Serializable, Iterator<B> {

    public TransformIterator(final Iterable<A> iterable, final Transform<A, B> transform) {
        this(iterable.iterator(), transform);
    }
    
    public TransformIterator(final Iterator<A> iterator,
            final Transform<A, B> transform) {
        this.iterator = iterator;
        this.transform = transform;
    }

    public Iterator<A> getIterator() {
        return iterator;
    }

    public void setIterator(final Iterator<A> iterator) {
        this.iterator = iterator;
    }

    public Transform<A, B> getTransform() {
        return transform;
    }

    public void setTransform(
            final Transform<A, B> transform) {
        this.transform = transform;
    }

    @Override public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override public B next() {
        return transform.transform(iterator.next());
    }

    public interface Transform<A, B> extends Serializable {
        B transform(A item);
    }
    
    private Iterator<A> iterator;
    private Transform<A, B> transform; 
}
