package tsml.classifiers.distance_based.utils.collections.params.distribution;

import java.util.Random;

/**
 * Purpose: model a distribution which can be sampled from, e.g. a normal distribution bell curve.
 * <p>
 * Contributors: goastler
 */

public abstract class BaseDistribution<A> implements Distribution<A> {

    public BaseDistribution() {
        
    }

    @Override public String toString() {
        return getClass().getSimpleName().replaceAll(Distribution.class.getSimpleName(), "");
    }
}
