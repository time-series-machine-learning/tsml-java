package tsml.classifiers.distance_based.knn.neighbour_iteration;

import tsml.classifiers.distance_based.knn.KNN;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import tsml.classifiers.distance_based.utils.collections.iteration.LinearListIterator;

import java.util.ArrayList;
import java.util.Iterator;

public class LinearNeighbourIteratorBuilder
    implements KNNLOOCV.NeighbourIteratorBuilder {

    protected KNNLOOCV knn;

    public LinearNeighbourIteratorBuilder(KNNLOOCV knn) {
        this.knn = knn;
    }

    public LinearNeighbourIteratorBuilder() {

    }

    public KNNLOOCV getKnn() {
        return knn;
    }

    public void setKnn(KNNLOOCV knn) {
        this.knn = knn;
    }

    @Override
    public Iterator<KNN.NeighbourSearcher> build() {
        if(knn == null) throw new IllegalStateException("knn not set");
        return new LinearListIterator<>(new ArrayList<>(knn.getSearchers()));
    }

}
