package utilities.cache;

import utilities.serialisation.SerialisedFunction;

import java.io.Serializable;
import java.util.HashMap;
import java.util.function.Function;

public class CachedFunction<I, O> implements Function<I, O>,
                                             Serializable {
    private final HashMap<I, O> cache = new HashMap<>();

    private SerialisedFunction<I, O> function;

    public CachedFunction(final SerialisedFunction<I, O> function) {
        this.function = function;
    }

    public O apply(I input) {
        return cache.computeIfAbsent(input, function);
    }

    public void clear() {
        cache.clear();
    }

    public SerialisedFunction<I, O> getFunction() {
        return function;
    }

    public void setFunction(final SerialisedFunction<I, O> function) {
        this.function = function;
    }

}
