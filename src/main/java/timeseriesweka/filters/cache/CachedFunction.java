package timeseriesweka.filters.cache;

import java.util.HashMap;
import java.util.function.Function;

public class CachedFunction<I, H, O> {
    private final HashMap<H, O> cache = new HashMap<>();

    private Function<I, O> function;
    private Function<I, H> hashFunction;

    public CachedFunction(final Function<I, O> function, final Function<I, H> hashFunction) {
        this.function = function;
        this.hashFunction = hashFunction;
    }

    public O get(I input) {
        H hash = hashFunction.apply(input);
        O result = cache.computeIfAbsent(hash, h -> function.apply(input));
        return result;
    }

    public void clear() {
        cache.clear();
    }

    public Function<I, O> getFunction() {
        return function;
    }

    public void setFunction(final Function<I, O> function) {
        this.function = function;
    }

    public Function<I, H> getHashFunction() {
        return hashFunction;
    }

    public void setHashFunction(final Function<I, H> hashFunction) {
        this.hashFunction = hashFunction;
    }
}
