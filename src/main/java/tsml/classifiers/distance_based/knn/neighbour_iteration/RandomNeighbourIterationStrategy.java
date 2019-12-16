package tsml.classifiers.distance_based.knn.neighbour_iteration;


import tsml.classifiers.distance_based.knn.KNN;
import tsml.classifiers.distance_based.knn.KNNCV;
import utilities.iteration.RandomIterator;

import java.util.ArrayList;
import java.util.Iterator;

public class RandomNeighbourIterationStrategy
    implements KNNCV.NeighbourIterationStrategy {


    @Override
    public Iterator<KNN.NeighbourSearcher> build(KNNCV knn) {
        if(knn == null) throw new IllegalStateException("no knn supplied");
        return new RandomIterator<>(knn.getSeed(), new ArrayList<>(knn.getSearchers()));
    }


}
