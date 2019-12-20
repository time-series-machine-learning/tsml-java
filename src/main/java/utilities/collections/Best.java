package utilities.collections;

public class Best<A extends Comparable<? super A>> {

    private final PrunedTreeMultiMap<A, A> map = new PrunedTreeMultiMap<A, A>(TreeMultiMap.newNaturalAsc(), 1);

    public void add(A item) {
        map.add(item, item);
    }

    public A get() {
        if(map.size() == 0) {
            return null;
        }
        return map.values().iterator().next().get(0);
    }

    public Best() {}

    public Best(A item) {
        add(item);
    }

}
