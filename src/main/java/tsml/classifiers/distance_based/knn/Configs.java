package tsml.classifiers.distance_based.knn;

import tsml.classifiers.distance_based.distances.Ddtw;
import tsml.classifiers.distance_based.distances.Dtw;
import tsml.classifiers.distance_based.knn.neighbour_iteration.LinearNeighbourIterationStrategy;
import tsml.classifiers.distance_based.knn.neighbour_iteration.RandomNeighbourIterationStrategy;
import utilities.ArrayUtilities;
import utilities.params.ParamSpace;
import weka.core.Instances;

import static utilities.ArrayUtilities.incrementalRange;

public class Configs {
    private Configs() {}

    public static KNNCV build1nnV1() {
        KNNCV classifier = new KNNCV();
        classifier.setEarlyAbandon(true);
        classifier.setK(1);
        classifier.setNeighbourLimit(-1);
        classifier.setNeighbourIterationStrategy(new LinearNeighbourIterationStrategy());
        classifier.setRandomTieBreak(false);
        return classifier;
    }

    public static KNNCV build1nnV2() {
        KNNCV classifier = new KNNCV();
        classifier.setEarlyAbandon(true);
        classifier.setK(1);
        classifier.setNeighbourLimit(-1);
        classifier.setNeighbourIterationStrategy(new RandomNeighbourIterationStrategy());
        classifier.setRandomTieBreak(true);
        return classifier;
    }


    public static KNNCV buildEd1nnV1() {
        KNNCV knn = build1nnV1();
        knn.setDistanceFunction(new Dtw(0));
        return knn;
    }

    public static KNNCV buildDtw1nnV1() {
        KNNCV knn = build1nnV1();
        knn.setDistanceFunction(new Dtw(-1));
        return knn;
    }

    public static KNNCV buildDdtw1nnV1() {
        KNNCV knn = build1nnV1();
        knn.setDistanceFunction(new Ddtw(-1));
        return knn;
    }

    public static KNNCV buildEd1nnV2() {
        KNNCV knn = build1nnV2();
        knn.setDistanceFunction(new Dtw(0));
        return knn;
    }

    public static KNNCV buildDtw1nnV2() {
        KNNCV knn = build1nnV2();
        knn.setDistanceFunction(new Dtw(-1));
        return knn;
    }

    public static KNNCV buildDdtw1nnV2() {
        KNNCV knn = build1nnV2();
        knn.setDistanceFunction(new Ddtw(-1));
        return knn;
    }
}
