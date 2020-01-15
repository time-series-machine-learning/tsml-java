package machine_learning.classifiers.tuned.incremental;

import java.io.Serializable;

public interface Optimiser extends Serializable {
    boolean shouldSource();
}
