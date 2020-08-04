package tsml.classifiers.distance_based.utils.collections.params.iteration;

import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceTest;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;

import java.util.Random;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomSearchIteratorTest {

    @Test
    public void testIteration() {
        RandomUtils.setSkipSingleElementPick(true);
        ParamSpace space = new ParamSpaceTest().build2DContinuousSpace();
        final int limit = 10;
        RandomSearchIterator iterator = new RandomSearchIterator(new Random(0), space, limit);
        iterator.setRandom(new ParamSpaceTest().buildRandom());
        StringBuilder stringBuilder = new StringBuilder();
        int count = 0;
        while(iterator.hasNext()) {
            count++;
            ParamSet paramSet = iterator.next();
            stringBuilder.append(paramSet);
            stringBuilder.append("\n");
        }
//        System.out.println(stringBuilder.toString());
        Assert.assertEquals(count, limit);
        Assert.assertEquals(stringBuilder.toString(),
            "-a, \"0.3654838936883285\", -b, \"0.6202682078357429\"\n"
                + "-a, \"0.31870871267505413\", -b, \"0.775218502558817\"\n"
                + "-a, \"0.2987726388986009\", -b, \"0.6666091997383249\"\n"
                + "-a, \"0.19259459237035925\", -b, \"0.9924207700999045\"\n"
                + "-a, \"0.43959125893624007\", -b, \"0.9706245897410573\"\n"
                + "-a, \"0.13747698301774242\", -b, \"0.5644485754368884\"\n"
                + "-a, \"0.07330082882325911\", -b, \"0.5116190612419447\"\n"
                + "-a, \"0.2733698785992328\", -b, \"0.9822434303384251\"\n"
                + "-a, \"0.052245343125485844\", -b, \"0.8125731817327797\"\n"
                + "-a, \"0.20539809774553086\", -b, \"0.8881561456374663\"\n"
        );
    }

}
