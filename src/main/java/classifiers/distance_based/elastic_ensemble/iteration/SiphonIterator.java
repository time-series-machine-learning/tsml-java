package classifiers.distance_based.elastic_ensemble.iteration;

public class SiphonIterator<A>
    extends AbstractIterator<A> {

    private AbstractIterator<A> source;
    private AbstractIterator<A> destination;

    public SiphonIterator(SiphonIterator<A> other) {
        this(other.source.iterator(), other.destination.iterator());
    }

    public SiphonIterator(AbstractIterator<A> source, AbstractIterator<A> destination) {
        this.source = source;
        this.destination = destination;
    }

    public AbstractIterator<A> getSource() {
        return source;
    }

    public void setSource(final AbstractIterator<A> source) {
        this.source = source;
    }

    public AbstractIterator<A> getDestination() {
        return destination;
    }

    public void setDestination(AbstractIterator<A> destination) {
        this.destination = destination;
    }

    @Override
    public SiphonIterator<A> iterator() {
        return new SiphonIterator<>(this);
    }

    @Override
    public boolean hasNext() {
        return source.hasNext();
    }

    @Override
    public A next() {
        A next = source.next();
        if (destination != null) {
            destination.add(next);
        }
        return next;
    }

    @Override
    public void remove() {
        source.remove();
    }

    @Override
    public void add(final A a) {
        source.add(a);
    }
}
