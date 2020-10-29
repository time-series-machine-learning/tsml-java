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
        ParamSpace space = new ParamSpaceTest().build2DContinuousSpace();
        final int limit = 10;
        RandomSearchIterator iterator = new RandomSearchIterator();
        iterator.setRandom(new Random(0));
        iterator.setIterationLimit(limit);
        iterator.buildSearch(space);
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
        Assert.assertEquals(
            "-a 0.4157204943935306 -b 0.8187087126750541\n" +
                    "-a 0.058503304403612566 -b 0.6666091997383249\n" +
                    "-a 0.3065178840223069 -b 0.9395912589362401\n" +
                    "-a 0.08798840101774008 -b 0.5644485754368884\n" +
                    "-a 0.35258737223772796 -b 0.7733698785992328\n" +
                    "-a 0.2814748369491896 -b 0.8125731817327797\n" +
                    "-a 0.00746354294055912 -b 0.9953613928573914\n" +
                    "-a 0.43383933414698683 -b 0.8665760350974969\n" +
                    "-a 0.006403325787859793 -b 0.7633497173024331\n" +
                    "-a 0.49233707140341276 -b 0.5415311991124574\n"
                , stringBuilder.toString()
        );
    }

}
