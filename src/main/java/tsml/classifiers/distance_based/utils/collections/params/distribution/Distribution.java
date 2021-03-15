package tsml.classifiers.distance_based.utils.collections.params.distribution;

import java.io.Serializable;
import java.util.Random;

public interface Distribution<A> extends Serializable {

    A sample(Random random);

}
