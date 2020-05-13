package tsml.classifiers.distance_based.proximity.splitting;

import java.util.List;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap;
import tsml.classifiers.distance_based.utils.random.RandomUtils;
import weka.core.Instance;
import weka.core.Instances;

public class BestOfNSplits extends Split {
    private int numSplits = 5;
    private Splitter splitter;
    private Split bestSplit;

    public BestOfNSplits(final Splitter splitter, final Random random) {
        this(splitter, random, 5);
    }

    public BestOfNSplits(final Splitter splitter, final Random random, final int numSplits) {
        super(random);
        setNumSplits(numSplits);
        setSplitter(splitter);
    }

    public int getNumSplits() {
        return numSplits;
    }

    public void setNumSplits(final int numSplits) {
        this.numSplits = numSplits;
    }

    public Splitter getSplitter() {
        return splitter;
    }

    public void setSplitter(final Splitter splitter) {
        Assert.assertNotNull(splitter);
        this.splitter = splitter;
    }

    @Override
    public List<Instances> performSplit(final Instances data) {
        final PrunedMultimap<Double, Split> map = PrunedMultimap.desc();
        map.setSoftLimit(1);
        for(int i = 0; i < getNumSplits(); i++) {
            Split split = splitter.buildSplit(data);
            split.buildSplit();
            double score = split.getScore();
            map.put(score, split);
        }
        map.hardPruneToSoftLimit();
        Split choice = RandomUtils.choice(map.values(), getRandom());
        setBestSplit(choice);
        return choice.getPartitions();
    }

    public Split getBestSplit() {
        return bestSplit;
    }

    public BestOfNSplits setBestSplit(final Split bestSplit) {
        this.bestSplit = bestSplit;
        return this;
    }

    @Override
    public int getPartitionIndexFor(final Instance instance) {
        return bestSplit.getPartitionIndexFor(instance);
    }

    @Override
    public double[] distributionForInstance(final Instance instance, final int partitionIndex) {
        return bestSplit.distributionForInstance(instance, partitionIndex);
    }
}
