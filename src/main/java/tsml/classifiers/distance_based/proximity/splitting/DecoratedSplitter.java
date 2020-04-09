package tsml.classifiers.distance_based.proximity.splitting;

import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class DecoratedSplitter implements Splitter {
    private final Splitter delegate;

    public DecoratedSplitter(final Splitter delegate) {
        this.delegate = delegate;
    }

    public Split buildSplit(final Instances data) {
        return delegate.buildSplit(data);
    }
}
