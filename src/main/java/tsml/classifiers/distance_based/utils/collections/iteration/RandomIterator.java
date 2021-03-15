package tsml.classifiers.distance_based.utils.collections.iteration;

import tsml.classifiers.distance_based.utils.system.random.Randomised;
import weka.core.Randomizable;

import java.io.Serializable;
import java.util.Random;

public interface RandomIterator<A> extends DefaultListIterator<A>, Serializable, Randomised {
    boolean withReplacement();

    void setWithReplacement(boolean withReplacement);
}
