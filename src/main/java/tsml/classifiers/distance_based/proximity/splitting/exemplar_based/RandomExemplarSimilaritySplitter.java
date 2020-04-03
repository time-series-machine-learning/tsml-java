package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import com.beust.jcommander.internal.Lists;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.proximity.splitting.Scorer;
import tsml.classifiers.distance_based.proximity.splitting.Splitter;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.classifiers.distance_based.utils.params.iteration.RandomSearchIterator;
import tsml.classifiers.distance_based.utils.random.RandomUtils;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomExemplarSimilaritySplitter extends Splitter {

    private List<ParamSpace> paramSpaces = new ArrayList<>();
    private Random random;
    private ExemplarPicker exemplarPicker;
    private List<RandomSearchIterator> randomSearchIteratorList = new ArrayList<>();
    private boolean useEarlyAbandon = true;
    private Scorer scorer = Scorer.getGiniImpurityScorer();

    public List<ParamSpace> getParamSpaces() {
        return paramSpaces;
    }

    protected List<RandomSearchIterator> getRandomSearchIteratorList() {
        return randomSearchIteratorList;
    }

    public RandomExemplarSimilaritySplitter setParamSpaces(
        List<ParamSpace> paramSpaces) {
        Assert.assertNotNull(paramSpaces);
        Assert.assertFalse(paramSpaces.isEmpty());
        this.paramSpaces = paramSpaces;
        randomSearchIteratorList = new ArrayList<>();
        Random random = getRandom();
        for(ParamSpace paramSpace : paramSpaces) {
            randomSearchIteratorList.add(new RandomSearchIterator(random, paramSpace));
        }
        return this;
    }

    public Random getRandom() {
        return random;
    }

    public RandomExemplarSimilaritySplitter setRandom(
        Random random) {
        Assert.assertNotNull(random);
        for(RandomSearchIterator iterator : randomSearchIteratorList) {
            iterator.setRandom(random);
        }
        this.random = random;
        return this;
    }

    public RandomExemplarSimilaritySplitter(List<ParamSpace> paramSpaces, Random random,
        ExemplarPicker exemplarPicker) {
        setExemplarPicker(exemplarPicker);
        setRandom(random);
        setParamSpaces(paramSpaces);
    }

    private DistanceFunction pickDistanceFunction() {
        Random random = getRandom();
        RandomSearchIterator iterator = RandomUtils.choice(getRandomSearchIteratorList(), random);
        Assert.assertTrue(iterator.hasNext());
        ParamSet paramSet = iterator.next();
        List<Object> list = paramSet.get(DistanceMeasureable.getDistanceFunctionFlag());
        Assert.assertTrue(list.size() == 1);
        Object obj = list.get(0);
        return (DistanceFunction) obj;
    }

    @Override
    public RandomExemplarSimilaritySplit buildSplit(Instances data) {
        List<List<Instance>> exemplars = exemplarPicker.pickExemplars(data);
        List<Instances> partitions = Lists.newArrayList(exemplars.size());
        for(List<Instance> exemplarGroup : exemplars) {
            partitions.add(new Instances(data, 0));
        }
        DistanceFunction distanceFunction = pickDistanceFunction();
        RandomExemplarSimilaritySplit split = new RandomExemplarSimilaritySplit(data, distanceFunction,
            exemplars, getRandom());
        split.setScorer(getScorer());
        return split;
    }

    public Scorer getScorer() {
        return scorer;
    }

    public RandomExemplarSimilaritySplitter setScorer(final Scorer scorer) {
        this.scorer = scorer;
        return this;
    }

    public ExemplarPicker getExemplarPicker() {
        return exemplarPicker;
    }

    public RandomExemplarSimilaritySplitter setExemplarPicker(
        ExemplarPicker exemplarPicker) {
        Assert.assertNotNull(exemplarPicker);
        this.exemplarPicker = exemplarPicker;
        return this;
    }

    public boolean isUseEarlyAbandon() {
        return useEarlyAbandon;
    }

    public RandomExemplarSimilaritySplitter setUseEarlyAbandon(final boolean useEarlyAbandon) {
        this.useEarlyAbandon = useEarlyAbandon;
        return this;
    }
}
