package tsml.classifiers.distance_based.utils.collections.box;

public class ImmutableBox<E> {

    protected E contents = null;

    public ImmutableBox(E contents) {
        set(contents);
    }

    public E get() {
        return contents;
    }

    protected void set(final E contents) {
        this.contents = contents;
    }

    @Override public String toString() {
        return contents.toString();
    }
}
