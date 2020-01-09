package tsml.classifiers.distance_based.knn;

import tsml.classifiers.distance_based.distances.Ddtw;
import tsml.classifiers.distance_based.distances.Dtw;
import tsml.classifiers.distance_based.knn.neighbour_iteration.LinearNeighbourIterationStrategy;
import tsml.classifiers.distance_based.knn.neighbour_iteration.RandomNeighbourIterationStrategy;

import static utilities.ArrayUtilities.incrementalRange;

public class KnnConfigs {
    private KnnConfigs() {}

    public static KnnLoocv build1nnV1() {
        KnnLoocv classifier = new KnnLoocv();
        classifier.setEarlyAbandon(true);
        classifier.setK(1);
        classifier.setNeighbourLimit(-1);
        classifier.setNeighbourIterationStrategy(new LinearNeighbourIterationStrategy());
        classifier.setRandomTieBreak(false);
        return classifier;
    }

    public static KnnLoocv build1nnV2() {
        KnnLoocv classifier = new KnnLoocv();
        classifier.setEarlyAbandon(true);
        classifier.setK(1);
        classifier.setNeighbourLimit(-1);
        classifier.setNeighbourIterationStrategy(new RandomNeighbourIterationStrategy());
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
