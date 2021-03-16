package tsml.classifiers.distance_based.utils.classifiers.contracting;

import tsml.classifiers.TestTimeContractable;

public interface ContractedTest extends TimedTest, TestTimeContractable {

    long getTestTimeLimit();

    default boolean hasTestTimeLimit() {
        return getTestTimeLimit() > 0;
    }

    default boolean insideTestTimeLimit(long nanos) {
        return !hasTestTimeLimit() || nanos < getTestTimeLimit();
    }

}
