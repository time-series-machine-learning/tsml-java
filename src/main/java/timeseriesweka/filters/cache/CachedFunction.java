package timeseriesweka.filters.cache;

import java.util.HashMap;
import java.util.function.Function;

public class CachedFunction<I, O> implements Function<I, O> {
    private final HashMap<I, O> cache = new HashMap<>();

    private Function<I, O> function;

    public CachedFunction(final Function<I, O> function) {
        this.function = function;
    }

    public O apply(I input) {
        O result = cache.computeIfAbsent(input, h -> function.apply(input));
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

}
