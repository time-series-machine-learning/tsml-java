package classifiers.tuning;


import java.util.Comparator;
import java.util.List;
import java.util.TreeMap;

public class Retainer<A, B> {
    private final TreeMap<A, List<B>> map;
    private final int max;

    public Retainer(Comparator<A> comparator, int max) {
        map = new TreeMap<>(comparator);
        this.max = max;
    }

    public void add(A benchmark, B value) {

    }
}
