package utilities;

import weka.core.Randomizable;

public interface DefaultRandomizable extends Randomizable {
    default void setSeed(int seed) {}
    default int getSeed() {
        throw new UnsupportedOperationException();
    }
}
