package tsml.classifiers.distance_based.utils.random;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomUtils {

    private RandomUtils() {}

    /**
     * this function gets the next seed from the rng, sources a random number from the rng and then sets the seed.
     * This ensures a sequence of seeds for each random number produced. This is helpful because usually you would
     * have to track how many calls / of what type you make to the rng if you want to find what the 100th random
     * number was. With this method, we set the seed each time before sourcing a number from the rng. These seeds can
     * be logged for direct access to the 100th random number, for example - which is much more convenient. The
     * result from this function contains both the sourced random number and the seed which has been applied to the
     * rng. Take this seed and apply it to a different rng with the same rng source operation (e.g. nextInt) will
     * return the same result.
     * @param <A>
     * @param random
     * @param func
     * @return
     */
    public static <A> RandomResult<A> getRandAndSwitchSeed(Random random,  Function<Random, A> func) {
        int seed = random.nextInt();
        random.setSeed(seed);
        A result = func.apply(random);
        return new RandomResult<A>(result, seed);
    }

    public static class RandomResult<A> {
        private final A result;
        private final int seed;

        public RandomResult(A result, int seed) {
            this.result = result;
            this.seed = seed;
        }

        public A getResult() {
            return result;
        }

        public int getSeed() {
            return seed;
        }
    }

    /**
     * randomly choose n elements from a list
     * @param list
     * @param random
     * @param numChoices
     * @param <A>
     * @return
     */
    public static <A> List<A> choice(List<A> list, Random random, int numChoices) {
        Assert.assertNotNull(random);
        Assert.assertNotNull(list);
        Assert.assertTrue(numChoices > 0);
        final List<A> choices = new ArrayList<>();
        final List<Integer> indices = CollectionUtils.sequence(list.size());
        for(int i = 0; i < numChoices; i++) {
            final int index = indices.remove(random.nextInt(indices.size()));
            final A value = list.get(index);
            choices.add(value);
        }
        return choices;
    }

    /**
     * randomly choose 1 element from  a list
     * @param list
     * @param random
     * @param <A>
     * @return
     */
    public static <A> A choice(List<A> list, Random random) {
        Assert.assertNotNull(random);
        Assert.assertNotNull(list);
        return list.get(random.nextInt(list.size()));
    }
}
