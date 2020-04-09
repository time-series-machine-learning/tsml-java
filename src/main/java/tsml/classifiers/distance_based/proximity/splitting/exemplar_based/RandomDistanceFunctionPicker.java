package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.classifiers.distance_based.utils.params.iteration.RandomSearchIterator;
import tsml.classifiers.distance_based.utils.random.RandomUtils;
import weka.core.DistanceFunction;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomDistanceFunctionPicker implements DistanceFunctionPicker {

    private Random random;
    private List<ParamSpace> paramSpaces = new ArrayList<>();
    private List<RandomSearchIterator> randomSearchIteratorList = new ArrayList<>();

    public RandomDistanceFunctionPicker(final Random random,
        final List<ParamSpace> paramSpaces) {
        setParamSpaces(paramSpaces);
        setRandom(random);
    }

    public RandomDistanceFunctionPicker setParamSpaces(List<ParamSpace> paramSpaces) {
        Assert.assertNotNull(paramSpaces);
        Assert.assertFalse(paramSpaces.isEmpty());
        this.paramSpaces = paramSpaces;
        randomSearchIteratorList = new ArrayList<>();
        Random random = getRandom();
        for(ParamSpace paramSpace : paramSpaces) {
            RandomSearchIterator iterator = new RandomSearchIterator(random, paramSpace);
            iterator.disableIterationLimit();
            randomSearchIteratorList.add(iterator);
        }
        return this;
    }


    @Override
    public DistanceFunction pickDistanceFunction() {
        Random random = getRandom();
        RandomSearchIterator iterator = RandomUtils.choice(getRandomSearchIteratorList(), random);
        Assert.assertTrue(iterator.hasNext());
        ParamSet paramSet = iterator.next();
        List<Object> list = paramSet.get(DistanceMeasureable.getDistanceFunctionFlag());
        Assert.assertEquals(1, list.size());
        Object obj = list.get(0);
        return (DistanceFunction) obj;
    }

    public List<ParamSpace> getParamSpaces() {
        return paramSpaces;
    }

    public List<RandomSearchIterator> getRandomSearchIteratorList() {
        return randomSearchIteratorList;
    }

    private RandomDistanceFunctionPicker setRandomSearchIteratorList(
        final List<RandomSearchIterator> randomSearchIteratorList) {
        Assert.assertNotNull(randomSearchIteratorList);
        this.randomSearchIteratorList = randomSearchIteratorList;
        return this;
    }

    public Random getRandom() {
        return random;
    }

    public RandomDistanceFunctionPicker setRandom(final Random random) {
        Assert.assertNotNull(random);
        this.random = random;
        for(RandomSearchIterator iterator : getRandomSearchIteratorList()) {
            iterator.setRandom(random);
        }
        return this;
    }
}
