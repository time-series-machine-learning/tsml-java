package classifiers.distance_based.elastic_ensemble;

public enum ParameterSpaceIterationStrategy {
    RANDOM,
    ROUND_ROBIN;

    public static ParameterSpaceIterationStrategy fromString(String str) {
        for (ParameterSpaceIterationStrategy s : ParameterSpaceIterationStrategy.values()) {
            if (s.name()
                 .equals(str)) {
                return s;
            }
        }
        throw new IllegalArgumentException("No enum value by the name of " + str);
    }
}
