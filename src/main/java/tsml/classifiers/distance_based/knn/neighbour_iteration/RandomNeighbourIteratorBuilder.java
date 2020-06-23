package tsml.classifiers.distance_based.knn.neighbour_iteration;


import tsml.classifiers.distance_based.knn.KNN;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;

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
        // set this to a random seed sourced from the knn's rng
        return new RandomIterator<>(knn.getRandom(), knn.getSearchers());
    }


}
