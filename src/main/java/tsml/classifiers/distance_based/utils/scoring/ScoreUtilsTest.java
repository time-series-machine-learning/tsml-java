package tsml.classifiers.distance_based.utils.scoring;

import com.beust.jcommander.internal.Lists;
import org.junit.Assert;
import org.junit.Test;
import utilities.Utilities;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ScoreUtilsTest {

    @Test
    public void testGiniImpurityEntropyImpure() {
        double score = ScoreUtils.giniImpurityEntropyFromClassCounts(Lists.newArrayList(5, 5));
        // System.out.println(score);
        Assert.assertTrue(score == 0.5);
    }

    @Test
    public void testGiniImpurityEntropyPure() {
        double score = ScoreUtils.giniImpurityEntropyFromClassCounts(Lists.newArrayList(10, 0));
        // System.out.println(score);
        Assert.assertTrue(score == 0);
    }

    @Test
    public void testGiniImpurityEntropyA() {
        double score = ScoreUtils.giniImpurityEntropyFromClassCounts(Lists.newArrayList(4, 6));
        // System.out.println(score);
        Assert.assertTrue(score == 0.48);
    }

    @Test
    public void testGiniImpurityEntropyB() {
        double score = ScoreUtils.giniImpurityEntropyFromClassCounts(Lists.newArrayList(1, 9));
        // System.out.println(score);
        Assert.assertTrue(score == 0.17999999999999994);
    }

    @Test
    public void testGiniImpurityPure() {
        double score = ScoreUtils.giniImpurity(Lists.newArrayList(8, 4), Lists.newArrayList(
            Lists.newArrayList(8, 0),
            Lists.newArrayList(0, 4)
        ));
        // System.out.println(score);
        Assert.assertTrue(score == 0.4444444444444444);
    }

    @Test
    public void testGiniImpurityImpure() {
        double score = ScoreUtils.giniImpurity(Lists.newArrayList(8, 4), Lists.newArrayList(
            Lists.newArrayList(4, 2),
            Lists.newArrayList(4, 2)
        ));
        // System.out.println(score);
        Assert.assertTrue(score == 0);
    }

    @Test
    public void testGiniImpurityA() {
        double score = ScoreUtils.giniImpurity(Lists.newArrayList(8, 4), Lists.newArrayList(
            Lists.newArrayList(7, 3),
            Lists.newArrayList(1, 1)
        ));
        // System.out.println(score);
        Assert.assertEquals(score, 0.011111111111111000, 0.0000000000000001);
    }

    @Test
    public void testGiniImpurityB() {
        double score = ScoreUtils.giniImpurity(Lists.newArrayList(8, 4), Lists.newArrayList(
            Lists.newArrayList(7, 1),
            Lists.newArrayList(1, 3)
        ));
        // System.out.println(score);
        Assert.assertTrue(score == 0.1736111111111111);
    }

    @Test
    public void testInfoGainEntropyImpure() {
        double score = ScoreUtils.infoGainEntropyFromClassCounts(Lists.newArrayList(5, 5));
        // System.out.println(score);
        Assert.assertTrue(score == 1);
    }

    @Test
    public void testInfoGainEntropyPure() {
        double score = ScoreUtils.infoGainEntropyFromClassCounts(Lists.newArrayList(10, 0));
        // System.out.println(score);
        Assert.assertTrue(score == 0);
    }

    @Test
    public void testInfoGainPure() {
        double score = ScoreUtils.infoGain(Lists.newArrayList(8, 4), Lists.newArrayList(
            Lists.newArrayList(8, 0),
            Lists.newArrayList(0, 4)
        ));
        // System.out.println(score);
        Assert.assertTrue(score == 0.9182958340544896);
    }

    @Test
    public void testInfoGainImpure() {
        double score = ScoreUtils.infoGain(Lists.newArrayList(8, 4), Lists.newArrayList(
            Lists.newArrayList(4, 2),
            Lists.newArrayList(4, 2)
        ));
        // System.out.println(score);
        Assert.assertTrue(score == 0);
    }

    @Test
    public void testInfoGainA() {
        double score = ScoreUtils.infoGain(Lists.newArrayList(8, 4), Lists.newArrayList(
            Lists.newArrayList(7, 3),
            Lists.newArrayList(1, 1)
        ));
        // System.out.println(score);
        Assert.assertEquals(score, 0.01722008469557898, 0.0000000000000001);
    }

    @Test
    public void testInfoGainB() {
        double score = ScoreUtils.infoGain(Lists.newArrayList(8, 4), Lists.newArrayList(
            Lists.newArrayList(7, 1),
            Lists.newArrayList(1, 3)
        ));
        // System.out.println(score);
        Assert.assertTrue(score == 0.28549349710171434);
    }
}
