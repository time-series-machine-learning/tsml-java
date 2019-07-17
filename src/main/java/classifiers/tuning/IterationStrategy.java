package classifiers.tuning;

public enum IterationStrategy {
    RANDOM,
    LINEAR;

    public static IterationStrategy fromString(String str) {
        for (IterationStrategy s : values()) {
            if (s.name()
                    .equals(str)) {
                return s;
            }
        }
        throw new IllegalArgumentException("No enum value by the name of " + str);
    }
}
