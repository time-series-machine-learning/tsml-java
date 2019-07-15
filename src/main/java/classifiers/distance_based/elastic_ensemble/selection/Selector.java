package classifiers.distance_based.elastic_ensemble.selection;

import java.util.List;
import java.util.Random;

public interface Selector<A> {
    boolean add(A candidate);
    List<A> getSelected();
    void setRandom(Random random);
    Random getRandom();
    void clear();
}
