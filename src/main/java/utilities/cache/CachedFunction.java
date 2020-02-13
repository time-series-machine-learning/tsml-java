package utilities.cache;

import utilities.serialisation.SerFunction;

import java.io.Serializable;
import java.util.HashMap;
import java.util.function.Function;

public class CachedFunction<I, O> implements Function<I, O>,
                                             Serializable {
    private final HashMap<I, O> cache = new HashMap<>();

    private SerFunction<I, O> function;

    public CachedFunction(final SerFunction<I, O> function) {
        this.function = function;
    }

    public O apply(I input) {
        return cache.computeIfAbsent(input, function);
    }

    public void clear() {
        cache.clear();
    }

    public SerFunction<I, O> getFunction() {
        return function;
    }

    public void setFunction(final SerFunction<I, O> function) {
        this.function = function;
    }

}
