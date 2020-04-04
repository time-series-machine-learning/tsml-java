package tsml.classifiers.distance_based.proximity.splitting;

import java.util.Random;
import org.junit.Assert;
import weka.core.Instances;

public abstract class AbstractSplitterBuilder implements SplitterBuilder {
    private Instances data;
    private Random randomSource;

    @Override
    public SplitterBuilder setRandom(Random randomSource) {
        Assert.assertNotNull(randomSource);
        this.randomSource = randomSource;
        return this;
    }

    @Override
    public SplitterBuilder setData(Instances data) {
        this.data = data;
        return this;
    }

    @Override
    public Instances getData() {
        return data;
    }

    @Override
    public Random getRandom() {
        return randomSource;
    }
}
