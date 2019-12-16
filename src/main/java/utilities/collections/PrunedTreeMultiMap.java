package utilities.collections;

import weka.core.Randomizable;

import java.io.Serializable;
import java.util.*;

public class PrunedTreeMultiMap<B, A> extends TreeMultiMap<B, A> implements Serializable,
                                                                            Randomizable { // todo implement
    // map interface

//    public static <A, B  extends Comparable<? super B>> PrunedTreeMultiMap<? extends A, B> bestAsc() { // todo
//        PrunedTreeMultiMap<? extends A, B> prunedTreeMultiMap = new PrunedTreeMultiMap<>(TreeMultiMap.newNaturalAsc());
//    }

    private int limit = -1;
    private boolean hardLimit = false;
    private final TreeMultiMap<B, A> map;
    private int seed = 0;
    private final Random random = new Random(seed);

    public TreeMultiMap<B, A> getMap() {
        return map;
    }

    public PrunedTreeMultiMap(TreeMultiMap<B, A> map) {
        super(map.decorated().comparator());
        this.map = map;
    }

    public PrunedTreeMultiMap(TreeMultiMap<B, A> map, int limit) {
        this(map);
        setLimit(limit);
    }

    public int getLimit() {
        return limit;
    }

    public void setLimit(final int limit) {
        this.limit = limit;
    }

    public boolean hasLimit() {
        return limit >= 0;
    }

    private void prune() {
        if(hasLimit()) {
            int diff = map.size() - limit;
            if(diff > 0) {
                Map.Entry<B, List<A>> entry = map.lastEntry();
                List<A> values = entry.getValue();
                while(diff > 0 && values != null && values.size() <= diff) {
                    entry = map.pollLastEntry();
                    diff -= entry.getValue().size();
                    entry = map.lastEntry();
                    if(entry != null) {
                        values = entry.getValue();
                    } else {
                        values = null;
                    }
                }
            }
            if(hardLimit) {
                diff = map.size() - limit;
                if(diff > 0) {
                    Map.Entry<B, List<A>> entry = map.lastEntry();
                    List<A> values = entry.getValue();
                    while(diff > 0) {
                        diff--;
                        values.remove(random.nextInt(values.size()));
                    }
                    if(values.isEmpty()) {
                        map.pollLastEntry();
                    }
                }
            }
        }
    }

    public void add(B key, A object) {
        map.put(key, object);
        prune();
    }

    public Collection<List<A>> values() {
        return map.values();
    }


    public Set<Map.Entry<B, List<A>>> entrySet() {
        return map.entrySet();
    }

    public boolean isHardLimit() {
        return hardLimit;
    }

    public void setHardLimit(final boolean hardLimit) {
        this.hardLimit = hardLimit;
    }

    @Override public void setSeed(final int seed) {
        this.seed = seed;
        random.setSeed(seed);
    }

    @Override public int getSeed() {
        return seed;
    }
}
