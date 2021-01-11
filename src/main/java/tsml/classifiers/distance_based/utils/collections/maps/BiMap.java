package tsml.classifiers.distance_based.utils.collections.maps;

import java.util.Map;

public class BiMap<A, B> {

    public BiMap(final Map<A, B> map, final Map<B, A> inverseMap) {
        this.map = new BiMapper<>(map, inverseMap);
        this.inverseMap = new BiMapper<>(inverseMap, map);
    }
    
    private final BiMapper<A, B> map;
    private final BiMapper<B, A> inverseMap;
    
    public Map<A, B> map() {
        return map;
    }
    
    public Map<B, A> inverseMap() {
        return inverseMap;
    }
    
}
