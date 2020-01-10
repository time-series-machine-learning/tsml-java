package tsml.classifiers.distance_based.knn.neighbour_iteration;

import tsml.classifiers.distance_based.knn.Knn;
import tsml.classifiers.distance_based.knn.KnnLoocv;
import utilities.iteration.LinearListIterator;

import java.util.ArrayList;
import java.util.Iterator;

public class LinearNeighbourIteratorBuilder
    implements KnnLoocv.NeighbourIteratorBuilder {

    protected KnnLoocv knn;

    public LinearNeighbourIteratorBuilder(KnnLoocv knn) {
        this.knn = knn;
    }

    public KnnLoocv getKnn() {
        return knn;
    }

    public void setKnn(KnnLoocv knn) {
        this.knn = knn;
    }

    @Override
    public Iterator<Knn.NeighbourSearcher> build() {
        return new LinearListIterator<>(new ArrayList<>(knn.getSearchers()));
    }

}
