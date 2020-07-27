package tsml.classifiers.distance_based.utils.collections.cache;

import java.io.Serializable;
import java.util.HashMap;
import java.util.function.Function;

/**
 * Purpose: class to cache the result of a function.
 * @param <I> the input type of the function.
 * @param <O> the output type of the function
 *
 * Contributors: goastler
 */
public class CachedFunction<I, O> implements Function<I, O>,
    Serializable {

    private final HashMap<I, O> cache = new HashMap<>();

    private Function<I, O> function;

    public CachedFunction(final Function<I, O> function) {
        this.function = function;
    }

    public O apply(I input) {
        return cache.computeIfAbsent(input, function);
    }

    public void clear() {
        cache.clear();
    }

    public Function<I, O> getFunction() {
        return function;
    }

    public void setFunction(Function<I, O> function) {
        if((function instanceof Serializable)) {
            function = (Serializable & Function<I, O>) function::apply;
        }
        this.function = function;
    }

}
