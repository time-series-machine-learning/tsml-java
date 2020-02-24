package tsml.classifiers;

public interface TestSeedable {

    default int getTestSeed() {
        return -1;
    }

    default void setTestSeed(int seed) {

    }
}
