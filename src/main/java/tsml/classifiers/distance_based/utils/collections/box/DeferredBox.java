package tsml.classifiers.distance_based.utils.collections.box;

import java.util.function.Supplier;

/**
 * Purpose: defer an operation until needed.
 * <p>
 * Contributors: goastler
 */
public class DeferredBox<A> {

    private final Supplier<A> getter;
    private A object;

    public DeferredBox(final Supplier<A> getter) {
        this.getter = getter;
    }

    public A get() {
        if(object == null) {
            object = getter.get();
        }
        return object;
    }

}
