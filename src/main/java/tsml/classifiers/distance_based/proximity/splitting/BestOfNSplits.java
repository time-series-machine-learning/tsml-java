package tsml.classifiers.distance_based.proximity.splitting;

import java.util.List;
import java.util.Random;
import tsml.classifiers.distance_based.proximity.splitting.Split;
import tsml.classifiers.distance_based.proximity.splitting.Splitter;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap;
import tsml.classifiers.distance_based.utils.random.RandomUtils;
import weka.core.Instance;
import weka.core.Instances;

public class BestOfNSplits extends Splitter {
    private int numSplits = 5;
    private Splitter splitter;
    private Random random;

    public BestOfNSplits(final Splitter splitter, final Random random) {
        this(5, splitter, random);
    }

    public BestOfNSplits(final int numSplits, final Splitter splitter, final Random random) {
        setNumSplits(numSplits);
        setSplitter(splitter);
        setRandom(random);
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
        this.splitter = splitter;
    }

    @Override
    public Split buildSplit(final Instances data) {
        final PrunedMultimap<Double, Split> map = PrunedMultimap.desc();
        map.setSoftLimit(1);
        for(int i = 0; i < getNumSplits(); i++) {
            Split split = splitter.buildSplit(data);
            split.findPartitions();
            double score = split.getScore();
            map.put(score, split);
        }
        // todo set random in pruned multimap?
        map.hardPruneToSoftLimit();
        Split choice = RandomUtils.choice(map.values(), getRandom());
        return choice;
    }

    public Random getRandom() {
        return random;
    }

    public void setRandom(final Random random) {
        this.random = random;
    }
}
