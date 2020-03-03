package tsml.classifiers.distance_based.knn.neighbour_iteration;


import tsml.classifiers.distance_based.knn.KNN;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import tsml.classifiers.distance_based.utils.iteration.RandomListIterator;

import java.util.Iterator;

public class RandomNeighbourIteratorBuilder
 extends LinearNeighbourIteratorBuilder {

    public RandomNeighbourIteratorBuilder() {}

    public RandomNeighbourIteratorBuilder(KNNLOOCV knn) {
        super(knn);
    }

    @Override
    public Iterator<KNN.NeighbourSearcher> build() {
        if(knn == null) throw new IllegalStateException("knn not set");
        return new RandomListIterator<>(knn.getSeed(), knn.getSearchers());
    }


}
