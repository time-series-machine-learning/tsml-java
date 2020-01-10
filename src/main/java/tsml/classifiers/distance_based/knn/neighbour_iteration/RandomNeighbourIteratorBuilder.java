package tsml.classifiers.distance_based.knn.neighbour_iteration;


import tsml.classifiers.distance_based.knn.Knn;
import tsml.classifiers.distance_based.knn.KnnLoocv;
import utilities.iteration.RandomIterator;

import java.util.Iterator;

public class RandomNeighbourIteratorBuilder
 extends LinearNeighbourIteratorBuilder {

    public RandomNeighbourIteratorBuilder(KnnLoocv knn) {
        super(knn);
    }

    @Override
    public Iterator<Knn.NeighbourSearcher> build() {
        return new RandomIterator<>(knn.getSeed(), knn.getSearchers());
    }


}
