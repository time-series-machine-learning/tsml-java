package tsml.classifiers.distance_based.utils.collections.box;

public class Box<E> extends ImmutableBox<E> {

    public Box() {
        super(null);
    }

    public Box(final E contents) {
        super(contents);
    }

    @Override public void set(final E contents) {
        super.set(contents);
    }

}
