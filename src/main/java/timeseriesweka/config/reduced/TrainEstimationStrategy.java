package timeseriesweka.config.reduced;

public enum TrainEstimationStrategy {
    RANDOM,
    LINEAR,
    ROUND_ROBIN_RANDOM,
    DISTRIBUTED_RANDOM;

    public static TrainEstimationStrategy fromString(String str) {
        for (TrainEstimationStrategy s : values()) {
            if (s.name()
                 .equals(str)) {
                return s;
            }
        }
        throw new IllegalArgumentException("No enum value by the name of " + str);
    }
}
