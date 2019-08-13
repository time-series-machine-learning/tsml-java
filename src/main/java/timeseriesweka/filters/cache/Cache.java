package timeseriesweka.filters.cache;

import java.util.HashMap;

public class Cache<A, B, C> {
    private final HashMap<A, HashMap<B, C>> cache = new HashMap<>();

    public C get(A firstKey, B secondKey) {
        HashMap<B, C> subCache = cache.get(firstKey);
        if(subCache != null) {
            return subCache.get(secondKey);
        }
        return null;
    }

    public void put(A firstKey, B secondkey, C value) {
        HashMap<B, C> subCache = cache.computeIfAbsent(firstKey, k -> new HashMap<>());
        subCache.put(secondkey, value);
    }

    public void clear() {
        cache.clear();
    }

    public void remove(A firstKey, B secondKey) {
        HashMap<B, C> subCache = cache.get(firstKey);
        if(subCache != null) {
            subCache.remove(secondKey);
            if(subCache.isEmpty()) {
                cache.remove(firstKey);
            }
        }
    }
}
