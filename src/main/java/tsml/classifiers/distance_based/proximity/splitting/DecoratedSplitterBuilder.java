package tsml.classifiers.distance_based.proximity.splitting;

import java.util.Random;
import weka.core.Instances;

public abstract class DecoratedSplitterBuilder implements SplitterBuilder {

    private SplitterBuilder splitterBuilder;

    public DecoratedSplitterBuilder(final SplitterBuilder splitterBuilder) {
        setSplitterBuilder(splitterBuilder);
    }

    public SplitterBuilder getDelegate() {
        return splitterBuilder;
    }

    public void setSplitterBuilder(final SplitterBuilder splitterBuilder) {
        this.splitterBuilder = splitterBuilder;
    }

    @Override
    public SplitterBuilder setRandom(final Random randomSource) {
        getDelegate().setRandom(randomSource);
        return this;
    }

    @Override
    public Random getRandom() {
        return getDelegate().getRandom();
    }

    @Override
    public SplitterBuilder setData(final Instances data) {
        getDelegate().setData(data);
        return this;
    }

    @Override
    public Instances getData() {
        return getDelegate().getData();
    }
}
