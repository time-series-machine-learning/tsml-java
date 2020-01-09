package tsml.classifiers.distance_based.knn.neighbour_iteration;


import tsml.classifiers.distance_based.knn.Knn;
import tsml.classifiers.distance_based.knn.KnnLoocv;
import utilities.iteration.RandomIterator;

import java.util.ArrayList;
import java.util.Iterator;

public class RandomNeighbourIterationStrategy
    implements KnnLoocv.NeighbourIterationStrategy {


    @Override
    public Iterator<Knn.NeighbourSearcher> build(KnnLoocv knn) {
        if(knn == null) throw new IllegalStateException("no knn supplied");
        return new RandomIterator<>(knn.getSeed(), new ArrayList<>(knn.getSearchers()));
    }


}
