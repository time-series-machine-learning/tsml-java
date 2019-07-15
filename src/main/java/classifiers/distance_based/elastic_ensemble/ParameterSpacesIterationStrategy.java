package classifiers.distance_based.elastic_ensemble;

public enum ParameterSpacesIterationStrategy {
    RANDOM,
    ROUND_ROBIN;

    public static ParameterSpacesIterationStrategy fromString(String str) {
        for (ParameterSpacesIterationStrategy s : ParameterSpacesIterationStrategy.values()) {
            if (s.name()
                 .equals(str)) {
                return s;
            }
        }
        throw new IllegalArgumentException("No enum value by the name of " + str);
    }
}
