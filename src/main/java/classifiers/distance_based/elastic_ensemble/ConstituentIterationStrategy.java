package classifiers.distance_based.elastic_ensemble;

public enum ConstituentIterationStrategy {
    RANDOM,
    ROUND_ROBIN;

    public static ConstituentIterationStrategy fromString(String str) {
        for (ConstituentIterationStrategy s : ConstituentIterationStrategy.values()) {
            if (s.name()
                 .equals(str)) {
                return s;
            }
        }
        throw new IllegalArgumentException("No enum value by the name of " + str);
    }
}
