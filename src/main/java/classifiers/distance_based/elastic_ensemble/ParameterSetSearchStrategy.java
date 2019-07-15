package classifiers.distance_based.elastic_ensemble;

public enum ParameterSetSearchStrategy {
    RANDOM,
    LINEAR,
    SPREAD;

    public static ParameterSetSearchStrategy fromString(String str) {
        for (ParameterSetSearchStrategy s : ParameterSetSearchStrategy.values()) {
            if (s.name()
                 .equals(str)) {
                return s;
            }
        }
        throw new IllegalArgumentException("No enum value by the name of " + str);
    }

}
