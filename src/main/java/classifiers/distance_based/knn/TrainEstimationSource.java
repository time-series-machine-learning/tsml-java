package classifiers.distance_based.knn;

public enum TrainEstimationSource {
    FROM_TRAIN_NEIGHBOURHOOD,
    FROM_TRAIN_SET;

    public static TrainEstimationSource fromString(String str) {
        for (TrainEstimationSource s : values()) {
            if (s.name()
                 .equals(str)) {
                return s;
            }
        }
        throw new IllegalArgumentException("No enum value by the name of " + str);
    }
}
