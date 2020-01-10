package tsml.classifiers.distance_based.knn;

import tsml.classifiers.distance_based.distances.Ddtw;
import tsml.classifiers.distance_based.distances.Dtw;
import tsml.classifiers.distance_based.knn.neighbour_iteration.LinearNeighbourIteratorBuilder;
import tsml.classifiers.distance_based.knn.neighbour_iteration.RandomNeighbourIteratorBuilder;

import static utilities.ArrayUtilities.incrementalRange;

public class KnnConfigs {
    private KnnConfigs() {}

    public static KnnLoocv build1nnV1() {
        KnnLoocv classifier = new KnnLoocv();
        classifier.setEarlyAbandon(true);
        classifier.setK(1);
        classifier.setNeighbourLimit(-1);
        classifier.setNeighbourIteratorBuilder(new LinearNeighbourIteratorBuilder(knn));
        classifier.setRandomTieBreak(false);
        return classifier;
    }

    public static KnnLoocv build1nnV2() {
        KnnLoocv classifier = new KnnLoocv();
        classifier.setEarlyAbandon(true);
        classifier.setK(1);
        classifier.setNeighbourLimit(-1);
        classifier.setNeighbourIteratorBuilder(new RandomNeighbourIteratorBuilder(knn));
        classifier.setRandomTieBreak(true);
        return classifier;
    }

    public static KnnLoocv buildEd1nnV1() {
        KnnLoocv knn = build1nnV1();
        knn.setDistanceFunction(new Dtw(0));
        return knn;
    }

    public static KnnLoocv buildDtw1nnV1() {
        KnnLoocv knn = build1nnV1();
        knn.setDistanceFunction(new Dtw(-1));
        return knn;
    }

    public static KnnLoocv buildDdtw1nnV1() {
        KnnLoocv knn = build1nnV1();
        knn.setDistanceFunction(new Ddtw(-1));
        return knn;
    }

    public static KnnLoocv buildEd1nnV2() {
        KnnLoocv knn = build1nnV2();
        knn.setDistanceFunction(new Dtw(0));
        return knn;
    }

    public static KnnLoocv buildDtw1nnV2() {
        KnnLoocv knn = build1nnV2();
        knn.setDistanceFunction(new Dtw(-1));
        return knn;
    }

    public static KnnLoocv buildDdtw1nnV2() {
        KnnLoocv knn = build1nnV2();
        knn.setDistanceFunction(new Ddtw(-1));
        return knn;
    }
}
