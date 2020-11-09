package tsml.classifiers.distance_based.utils.collections.iteration;

import java.io.Serializable;
import java.util.Random;

public interface RandomIterator<A> extends DefaultListIterator<A>, Serializable {
    boolean withReplacement();

    void setWithReplacement(boolean withReplacement);

    void setRandom(Random random);

    Random getRandom();
    
}
