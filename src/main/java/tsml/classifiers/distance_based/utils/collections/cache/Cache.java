package tsml.classifiers.distance_based.utils.collections.cache;

import java.util.HashMap;
import java.util.Map;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class Cache<A, B> extends Cached {

    private final Map<A, B> map = new HashMap<>();

    public B get(A key) {
        if(isRead()) {
            return map.get(key);
        } else {
            return null;
        }
    }

    public void put(A key, B value) {
        if(isWrite()) {
            map.put(key, value);
        }
    }

}
