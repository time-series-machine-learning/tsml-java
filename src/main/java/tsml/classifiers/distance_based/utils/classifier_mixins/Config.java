package tsml.classifiers.distance_based.utils.classifier_mixins;

import java.io.Serializable;

public abstract class Config<A> implements Serializable {
    public abstract <B extends A> B applyConfigTo(B classifier);
}
