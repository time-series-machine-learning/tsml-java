package classifiers.template.configuration.reduced;

public enum TrainEstimationSource {
    FROM_FULL_TRAIN_SET,
    FROM_REDUCED_TRAIN_SET;

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
