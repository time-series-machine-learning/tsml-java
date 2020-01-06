package utilities.collections;

import java.util.Comparator;

public class Best<A extends Comparable<? super A>> {

    private final PrunedTreeMultiMap<A, A> map;

    public void add(A item) {
        map.add(item, item);
    }

    public A get() {
        if(map.size() == 0) {
            return null;
        }
        return map.values().iterator().next().get(0);
    }

    public Best() {
        map = new PrunedTreeMultiMap<A, A>(TreeMultiMap.newNaturalDesc(), 1);
    }

    public Best(Comparator<A> comparator) {
        map = new PrunedTreeMultiMap<A, A>(new TreeMultiMap<>(comparator), 1);
    }

    public Best(A item) {
        this();
        add(item);
    }
}
