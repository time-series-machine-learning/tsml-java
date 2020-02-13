package tsml.classifiers;

import weka.core.Randomizable;

public interface TrainSeedable extends Randomizable {
    default int getTrainSeed() {
        return getSeed();
    }

    default void setTrainSeed(int seed) {
        setSeed(seed);
    }

}
