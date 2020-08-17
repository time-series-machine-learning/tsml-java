package tsml.classifiers.distance_based.utils.collections.cache;

import java.io.Serializable;
import java.util.HashMap;
import java.util.function.BiFunction;
import java.util.function.Supplier;

public class BiCache<A, B, C> extends Cached implements Serializable {

    // todo cache state read / write

    private final HashMap<A, HashMap<B, C>> cache = new HashMap<>();

    public C getAndPut(A firstKey, B secondKey, Supplier<C> supplier) {
        C result = get(firstKey, secondKey);
        if(result == null) {
            result = supplier.get();
        }
        put(firstKey, secondKey, result);
        return result;
    }

    public C get(A firstKey, B secondKey) {
        C result = null;
        HashMap<B, C> subCache = cache.get(firstKey);
        if(subCache != null) {
            result = subCache.get(secondKey);
        }
        return result;
    }

    public void put(A firstKey, B secondkey, C value) {
        HashMap<B, C> subCache = cache.computeIfAbsent(firstKey, k -> new HashMap<>());
        subCache.put(secondkey, value);
    }

    public boolean contains(A firstKey, B secondKey) {
        return get(firstKey, secondKey) != null;
    }

    public void clear() {
        cache.clear();
    }

    public boolean remove(A firstKey, B secondKey) {
        HashMap<B, C> subCache = cache.get(firstKey);
        if(subCache != null) {
            C removed = subCache.remove(secondKey);
            if(subCache.isEmpty()) {
                cache.remove(firstKey);
            }
            return removed != null;
        }
        return false;
    }

    public C computeIfAbsent(A firstKey, B secondKey, BiFunction<A, B, C> function) {
        C result = get(firstKey, secondKey);
        if(result == null) {
            result = function.apply(firstKey, secondKey);
            put(firstKey, secondKey, result);
        }
        return result;
    }
}
