package tsml.classifiers.distance_based.utils.classifiers.contracting;

import tsml.classifiers.TestTimeContractable;
import tsml.classifiers.distance_based.utils.classifiers.TestTimeable;

public interface ContractedTest extends TestTimeable, TestTimeContractable {

    long getTestTimeLimit();

    default boolean hasTestTimeLimit() {
        return getTestTimeLimit() > 0;
    }

    default boolean insideTestTimeLimit(long nanos) {
        return !hasTestTimeLimit() || nanos < getTestTimeLimit();
    }

}
