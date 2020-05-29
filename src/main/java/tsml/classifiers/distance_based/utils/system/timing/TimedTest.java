package tsml.classifiers.distance_based.utils.system.timing;

import tsml.classifiers.distance_based.utils.classifier_mixins.TestTimeable;

public interface TimedTest extends TestTimeable {
    StopWatch getTestTimer();

    default long getTestTime() {
        return getTestTimer().getTime();
    }
}
