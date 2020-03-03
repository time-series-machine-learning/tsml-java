package tsml.classifiers.distance_based.knn.neighbour_iteration;


import tsml.classifiers.distance_based.knn.Knn;
import tsml.classifiers.distance_based.knn.KnnLoocv;
import tsml.classifiers.distance_based.utils.iteration.RandomListIterator;

import java.util.Iterator;

public class RandomNeighbourIteratorBuilder
 extends LinearNeighbourIteratorBuilder {

    public RandomNeighbourIteratorBuilder() {}

    public RandomNeighbourIteratorBuilder(KnnLoocv knn) {
        super(knn);
    }

    @Override
    public Iterator<Knn.NeighbourSearcher> build() {
        if(knn == null) throw new IllegalStateException("knn not set");
        return new RandomListIterator<>(knn.getSeed(), knn.getSearchers());
    }


}
