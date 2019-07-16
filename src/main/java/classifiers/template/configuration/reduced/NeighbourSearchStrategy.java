package classifiers.template.configuration.reduced;

public enum NeighbourSearchStrategy {
    RANDOM,
    LINEAR,
    ROUND_ROBIN_RANDOM,
    DISTRIBUTED_RANDOM;

    public static NeighbourSearchStrategy fromString(String str) {
        for (NeighbourSearchStrategy s : NeighbourSearchStrategy.values()) {
            if (s.name()
                 .equals(str)) {
                return s;
            }
        }
        throw new IllegalArgumentException("No enum value by the name of " + str);
    }
}
