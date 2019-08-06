package timeseriesweka.filters;

import utilities.GenericTools;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.HashMap;
import java.util.function.Function;

public class Cache<A, B> {
    private final HashMap<A, B> cache = new HashMap<>();

    private final Function<A, B> function;

    public Cache(final Function<A, B> function) {this.function = function;}

    public B get(A value) {
        B result = cache.computeIfAbsent(value, function);
        return result;
    }
}
